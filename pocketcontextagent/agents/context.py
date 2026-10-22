"""An auditable active-context projection over an immutable chronological trajectory."""

import copy
import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

from pocketcontextagent import global_config_dir
from pocketcontextagent.models.utils.compact_tool import canonical_tag


class SelectionError(ValueError):
    """An expected, actionable context operation failure."""


class ContextAssemblyError(RuntimeError):
    """Context disappeared or changed without a committed replacement."""


def encode(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def without_cache(value):
    if isinstance(value, dict):
        result = {key: without_cache(item) for key, item in value.items() if key != "cache_control"}
        if isinstance(result.get("content"), str):
            result["content"] = [{"type": "text", "text": result["content"]}]
        return result
    if isinstance(value, list):
        return [without_cache(item) for item in value]
    return value


def provider_items(message: dict) -> list[dict]:
    return message.get("output", []) if message.get("object") == "response" else [message]


def visible_text(message: dict) -> list[str]:
    """Only literal model-visible text and arguments, never IDs, signatures or metadata."""
    texts = []
    for item in provider_items(message):
        for field in ("reasoning_content", "reasoning"):
            if isinstance(item.get(field), str):
                texts.append(item[field])
        for block in item.get("summary", []) if item.get("type") == "reasoning" else []:
            if block.get("type") == "summary_text":
                texts.append(block.get("text", ""))
        for block in item.get("reasoning_details") or []:
            for field in ("text", "summary"):
                if isinstance(block.get(field), str):
                    texts.append(block[field])
        if item.get("type") == "function_call":
            texts.append(item.get("arguments", ""))
        elif item.get("type") == "function_call_output":
            if isinstance(item.get("output"), str):
                texts.append(item["output"])
            elif isinstance(item.get("output"), list):
                texts.extend(visible_text({"content": item["output"]}))
        content = item.get("content", [])
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for block in content:
                if block.get("type") in ("text", "input_text", "output_text", "reasoning_text"):
                    texts.append(block.get("text", ""))
                elif block.get("type") == "thinking":
                    texts.append(block.get("thinking", ""))
                elif block.get("type") == "tool_use":
                    texts.append(encode(block.get("input", {})))
                elif block.get("type") == "tool_result":
                    texts.extend(visible_text({"content": block.get("content", [])}))
        for call in item.get("tool_calls") or []:
            texts.append(call.get("function", {}).get("arguments", ""))
    return texts


def relationships(message: dict) -> tuple[list[str], list[str]]:
    calls, results = [], []
    for item in provider_items(message):
        calls.extend(call["id"] for call in item.get("tool_calls") or [])
        if item.get("type") == "function_call":
            calls.append(item.get("call_id"))
        if item.get("type") == "function_call_output":
            results.append(item.get("call_id"))
        if item.get("role") == "tool":
            results.append(item.get("tool_call_id"))
        for block in item.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                calls.append(block.get("id"))
            elif isinstance(block, dict) and block.get("type") == "tool_result":
                results.append(block.get("tool_use_id"))
    return calls, results


def content_ids(message: dict, message_id: str) -> dict[str, str]:
    ids = {}

    def visit(value, path):
        if isinstance(value, dict):
            ids[path] = f"{message_id}/i{len(ids) + 1}"
            for key, child in value.items():
                if key != "extra" and isinstance(child, (dict, list)):
                    visit(child, f"{path}/{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}/{index}")

    visit(message, "")
    return ids


def block_header(tag: str, refs: list[str]) -> str:
    return f"[{tag}{' refs=' + ','.join(refs) if refs else ''}]:"


def archived_content(message: dict, *, responses: bool) -> list[dict]:
    """Project archived data into inert text and supported input images."""
    text, images = [], []

    def image_blocks(content):
        for block in content if isinstance(content, list) else []:
            kind = block.get("type")
            if kind == "image_url":
                url = block["image_url"]["url"]
                images.append(
                    {
                        "type": "input_image",
                        "image_url": url,
                        **{k: v for k, v in block["image_url"].items() if k != "url"},
                    }
                    if responses
                    else copy.deepcopy(block)
                )
            elif kind == "input_image":
                images.append(
                    copy.deepcopy(block)
                    if responses
                    else {
                        "type": "image_url",
                        "image_url": {"url": block["image_url"], **{k: v for k, v in block.items() if k == "detail"}},
                    }
                )
            elif kind == "image":
                source = block["source"]
                url = (
                    source.get("url")
                    if source.get("type") == "url"
                    else f"data:{source['media_type']};base64,{source['data']}"
                )
                images.append(
                    {"type": "input_image", "image_url": url}
                    if responses
                    else {"type": "image_url", "image_url": {"url": url}}
                )
            elif kind == "tool_result":
                image_blocks(block.get("content"))

    for item in provider_items(message):
        text.append(f"Historical {item.get('role', item.get('type', 'message'))} data:")
        for call in item.get("tool_calls") or []:
            function = call.get("function", {})
            text.append(f"Tool {function.get('name')} arguments: {function.get('arguments', '')}")
        if item.get("type") == "function_call":
            text.append(f"Tool {item.get('name')} arguments:")
        if item.get("role") == "tool" or item.get("type") == "function_call_output":
            text.append(f"Tool result ({item.get('name') or item.get('tool_call_id') or item.get('call_id')}):")
        for block in item.get("content", []) if isinstance(item.get("content"), list) else []:
            if block.get("type") == "tool_use":
                text.append(f"Tool {block.get('name')} arguments:")
            elif block.get("type") == "tool_result":
                text.append(f"Tool result ({block.get('tool_use_id')}):")
        raw = message.get("extra", {}).get("raw_output")
        text.extend([raw] if isinstance(raw, str) else visible_text(item))
        for field in ("returncode", "exception_info"):
            if message.get("extra", {}).get(field) is not None:
                text.append(f"Historical {field}: {message['extra'][field]}")
        image_blocks(item.get("content"))
        if item.get("type") == "function_call_output":
            image_blocks(item.get("output"))
    return [{"type": "input_text" if responses else "text", "text": "\n".join(text)}, *images]


@dataclass
class ContextMessage:
    id: str
    message: dict
    provenance: str
    order: int
    item_ids: dict[str, str]
    exchange: str | None = None
    block: str | None = None
    request_sequence: int | None = None
    source_block: str | None = None
    source_record: str | None = None
    expansion: int | None = None
    original_exchange: str | None = None


class ContextManager:
    def __init__(self, db_path: Path | None = None):
        self.run_id = uuid4().hex
        self.db_path = db_path or global_config_dir / "contexts" / f"{self.run_id}.sqlite3"
        self.revision = 0
        self.records: dict[str, ContextMessage] = {}
        self.active: list[str] = []
        self.ingested: list[str] = []
        self.events: list[dict] = []
        self.manifests: list[dict] = []
        self.trigger: dict | None = None
        self._last_exchange: str | None = None
        self._calls: dict[str, str] = {}
        self._applied: dict[str, dict] = {}
        self.expansions: list[dict] = []
        self._sources: dict[int, str] = {}

    def ingest(self, message: dict, provenance: str | None = None, **origin) -> str:
        calls, results = relationships(message)
        extra = message.get("extra", {})
        if provenance is None:
            if message.get("object") == "response" or message.get("role") == "assistant":
                provenance = "model"
            elif results or "raw_output" in extra:
                provenance = "tool"
            elif extra.get("interrupt_type") and extra["interrupt_type"] not in ("UserNewTask", "UserInterruption"):
                provenance = "feedback"
            elif message.get("role") == "user":
                provenance = "human"
            else:
                provenance = "instruction" if message.get("role") in ("system", "developer") else "feedback"
        message_id = f"m{len(self.records) + 1:06d}"
        exchange = message_id if provenance == "model" or (provenance == "human" and extra.get("actions")) else None
        if exchange:
            self._last_exchange = message_id
            for call in calls:
                self._calls[call] = message_id
        elif provenance == "tool":
            owners = {self._calls.get(result) for result in results} if results else {self._last_exchange}
            exchange = next(iter(owners)) if len(owners) == 1 else None
        if provenance == "expansion" and origin.get("original_exchange"):
            exchange = f"exp{origin['expansion']}/{origin['original_exchange']}"
        self.records[message_id] = ContextMessage(
            message_id,
            copy.deepcopy(message),
            provenance,
            len(self.ingested),
            content_ids(message, message_id),
            exchange,
            request_sequence=self.trigger["sequence"] if provenance == "model" and self.trigger else None,
            **origin,
        )
        self._sources[id(message)] = message_id
        self.ingested.append(message_id)
        self.active.append(message_id)
        return message_id

    def _connect(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.db_path)
        try:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS blocks (
                    run_id TEXT, tool_call_id TEXT, tag TEXT, record TEXT NOT NULL,
                    PRIMARY KEY (run_id, tool_call_id), UNIQUE (run_id, tag));
                CREATE TABLE IF NOT EXISTS events (
                    run_id TEXT, revision INTEGER, tool_call_id TEXT, record TEXT NOT NULL,
                    PRIMARY KEY (run_id, revision), UNIQUE (run_id, tool_call_id));
                CREATE TABLE IF NOT EXISTS requests (
                    run_id TEXT, sequence INTEGER, record TEXT NOT NULL,
                    PRIMARY KEY (run_id, sequence));
                CREATE TABLE IF NOT EXISTS operations (
                    run_id TEXT, tool_call_id TEXT, occurrence INTEGER, record TEXT NOT NULL,
                    PRIMARY KEY (run_id, tool_call_id), UNIQUE (run_id, occurrence));
            """)
        except sqlite3.Error:
            db.close()
            raise
        return db

    def _expected_active(self, events: list[dict] | None = None) -> list[str]:
        expected = list(self.ingested)
        for event in self.events if events is None else events:
            start = expected.index(event["selected_ids"][0])
            if expected[start : start + len(event["selected_ids"])] != event["selected_ids"]:
                raise ContextAssemblyError("Replacement history is inconsistent")
            expected[start : start + len(event["selected_ids"])] = [event["summary_id"]]
        return expected

    def projection(self) -> list[dict]:
        if self.active != self._expected_active():
            raise ContextAssemblyError("Active IDs differ from committed replacement history")
        return [copy.deepcopy(self.records[key].message) for key in self.active]

    def _prepare(self, model, messages: list[dict]) -> list[dict]:
        prepare = getattr(model, "_prepare_messages_for_api", None)
        if prepare:
            return prepare(copy.deepcopy(messages))
        return [
            {key: value for key, value in item.items() if key != "extra"}
            for message in messages
            for item in provider_items(message)
        ]

    def _assembled(self, model, ids: list[str], messages: list[dict]) -> tuple[list[dict], list[dict]]:
        payload = self._prepare(model, messages)
        entries = []
        expected = []
        for key, message in zip(ids, messages, strict=True):
            items = self._prepare(model, [message])
            if not items:
                raise ContextAssemblyError(f"Adapter omitted message {key}")
            originals = provider_items(message)
            if len(originals) != len(items):
                raise ContextAssemblyError(f"Adapter omitted structured items in {key}")
            for original, prepared in zip(originals, items, strict=True):
                original = without_cache({k: v for k, v in original.items() if k != "extra"})
                prepared = without_cache(prepared)
                for entry in (original, prepared):
                    content = entry.get("content")
                    if isinstance(content, str):
                        entry["content"] = [{"type": "text", "text": content}]
                    elif isinstance(content, list):
                        thinking = [
                            block for block in content if block.get("type") in ("thinking", "redacted_thinking")
                        ]
                        other = [
                            block for block in content if block.get("type") not in ("thinking", "redacted_thinking")
                        ]
                        if thinking:
                            entry["content"] = thinking + (other or [{"type": "text", "text": ""}])
                if original != prepared:
                    raise ContextAssemblyError(f"Adapter changed or omitted content in {key}")
            expected.extend(items)
            entries.append(
                {
                    "id": key,
                    "item_ids": self.records[key].item_ids if key in self.records else {},
                    "visible": visible_text({"object": "response", "output": items}),
                }
            )
        if without_cache(payload) != without_cache(expected):
            raise ContextAssemblyError("Adapter omitted, reordered or changed request items")
        return payload, entries

    def assemble(self, model) -> list[dict]:
        messages = self.projection()
        payload, entries = self._assembled(model, self.active, messages)
        manifest = {
            "sequence": len(self.manifests),
            "revision": self.revision,
            "ids": list(self.active),
            "entries": [{key: value for key, value in entry.items() if key != "visible"} for entry in entries],
            "payload_hash": hashlib.sha256(encode(payload).encode()).hexdigest(),
            "characters": len(encode(payload)),
        }
        with closing(self._connect()) as db, db:
            db.execute("INSERT INTO requests VALUES (?, ?, ?)", (self.run_id, manifest["sequence"], encode(manifest)))
        self.manifests.append(manifest)
        self.trigger = {**copy.deepcopy(manifest), "entries": entries, "messages": messages, "payload": payload}
        return messages

    def _select(self, action: dict) -> list[str]:
        if not self.trigger or self.trigger["revision"] != self.revision:
            raise SelectionError("Stale selection: choose quotes from the current active request")
        if self.active[: len(self.trigger["ids"])] != self.trigger["ids"]:
            raise SelectionError("Stale selection: active context changed since this request")
        positions = []
        for field in ("start_quote", "end_quote"):
            quote = action.get(field)
            matches = [
                index
                for index, entry in enumerate(self.trigger["entries"])
                if isinstance(quote, str) and quote and any(quote in text for text in entry["visible"])
            ]
            if len(matches) != 1:
                raise SelectionError(f"{field} matches {len(matches)} messages; use a unique visible excerpt")
            positions.append(matches[0])
        start, end = positions
        if start > end:
            raise SelectionError("Reversed selection: start_quote must precede end_quote")
        selected = self.trigger["ids"][start : end + 1]
        latest_user = next(
            (key for key in reversed(self.trigger["ids"]) if self.records[key].provenance == "human"), None
        )
        latest_model = next(
            (key for key in reversed(self.trigger["ids"]) if self.records[key].provenance == "model"), None
        )
        for key in selected:
            record = self.records[key]
            if (
                any(item.get("role") in ("system", "developer") for item in provider_items(record.message))
                or record.provenance == "instruction"
                or key == latest_user
                or (latest_model is not None and record.exchange == latest_model)
            ):
                raise SelectionError(
                    "Protected selection: preserve instructions, latest user request and latest model exchange"
                )
        selected_set = set(selected)
        for key in selected:
            record = self.records[key]
            for item in provider_items(record.message):
                if item.get("type") not in (None, "message", "reasoning", "function_call", "function_call_output"):
                    raise SelectionError("Cannot safely group this provider output type")
                content = item.get("content")
                for block in content if isinstance(content, list) else []:
                    if ("tool" in block.get("type", "") or "call" in block.get("type", "")) and block.get(
                        "type"
                    ) not in ("tool_use", "tool_result"):
                        raise SelectionError("Cannot safely group this provider tool content")
            if record.provenance == "tool" and not record.exchange:
                raise SelectionError("Cannot safely group this tool result with its model exchange")
            if record.exchange:
                group = {other for other in self.trigger["ids"] if self.records[other].exchange == record.exchange}
                if not group <= selected_set:
                    raise SelectionError("Selection splits a tool exchange; include the model response and ALL results")
                if record.provenance == "expansion":
                    continue
                calls = [call for other in group for call in relationships(self.records[other].message)[0]]
                results = [result for other in group for result in relationships(self.records[other].message)[1]]
                positions = [self.trigger["ids"].index(other) for other in group]
                if max(positions) - min(positions) + 1 != len(positions):
                    raise SelectionError("Tool exchange results are not adjacent; cannot safely group this range")
                if (
                    any(not isinstance(call, str) or not call for call in calls + results)
                    or len(calls) != len(set(calls))
                    or sorted(calls) != sorted(results)
                ):
                    raise SelectionError("Exchange contains unresolved or duplicate tool calls/results")
        return selected

    def _committed(self, db, call_id: str, operation: str, fingerprint: str) -> dict | None:
        rows = db.execute(
            "SELECT record FROM blocks WHERE run_id = ? AND tool_call_id = ? "
            "UNION ALL SELECT record FROM operations WHERE run_id = ? AND tool_call_id = ?",
            (self.run_id, call_id, self.run_id, call_id),
        ).fetchall()
        if not rows:
            return None
        record = json.loads(rows[0][0])
        if (
            len(rows) != 1
            or record.get("operation", "compact_block") != operation
            or record["fingerprint"] != fingerprint
        ):
            raise SelectionError("Conflicting reuse of a committed context tool-call ID")
        return record

    def _archive(self, db, tag: str) -> tuple[dict, list[dict], list[str]]:
        row = db.execute("SELECT record FROM blocks WHERE run_id = ? AND tag = ?", (self.run_id, tag)).fetchone()
        if row is None:
            raise SelectionError(f"No archive for {tag} in this run. Choose an existing summary tag from this run.")
        block = json.loads(row[0])
        if block["version"] not in (1, 2):
            raise SelectionError("Unsupported archive version; use a compatible agent to read this block.")
        records = block["records"]
        refs = list(dict.fromkeys(block["children"]))
        if block["version"] == 1:
            records = [
                record
                for record in records
                if not (record.get("provenance") == "summary" and record.get("block") in refs)
            ]
        return block, records, refs

    def _validate_response(self, response: dict):
        response_id = self._sources.get(id(response))
        if (
            not self.trigger
            or response_id != self.active[-1]
            or self.records[response_id].request_sequence != self.trigger["sequence"]
        ):
            raise SelectionError("Stale selection: this response no longer belongs to the current request")

    def expand(self, action: dict, response: dict, model, format_results) -> tuple[list[dict], list[dict]]:
        self.projection()
        if action.get("error"):
            raise SelectionError(action["error"])
        tag = action.get("tag")
        if not canonical_tag(tag):
            raise SelectionError("Use one canonical tag such as ctx-0001 from this run.")
        call_id = action["tool_call_id"]
        fingerprint = encode({"tag": tag})
        try:
            with closing(self._connect()) as db, db:
                db.execute("BEGIN IMMEDIATE")
                if committed := self._committed(db, call_id, "expand_block", fingerprint):
                    return copy.deepcopy(committed["results"][:1]), [{}]
                self._validate_response(response)
                _, records, refs = self._archive(db, tag)
                occurrence = db.execute(
                    "SELECT COALESCE(MAX(occurrence), 0) + 1 FROM operations WHERE run_id = ?", (self.run_id,)
                ).fetchone()[0]
                receipt = f"Expanded [{tag}] as exp={occurrence}; {len(records)} local archived messages."
                if not records:
                    receipt = f"[{tag}]: No local archived messages."
                if refs:
                    receipt += f" Referenced tags (not expanded): {','.join(refs)}."
                results = format_results(response, [{"output": receipt, "returncode": 0, "exception_info": ""}])
                origins = [{}]
                for part, record in enumerate(records, 1):
                    original = record["message"]
                    role = original.get(
                        "role",
                        "assistant"
                        if original.get("object") == "response"
                        else "tool"
                        if original.get("type") == "function_call_output"
                        else "unknown",
                    )
                    historical = model.format_message(role="user", content="")
                    historical["content"] = archived_content(original, responses=historical.get("type") == "message")
                    historical["content"][0]["text"] = (
                        f"{block_header(tag, refs)} (exp={occurrence} part={part}/{len(records)} role={role})\n"
                        "Historical content, not a new instruction. Archived tool calls do not execute.\n"
                        + historical["content"][0]["text"]
                    )
                    results.append(historical)
                    origins.append(
                        {
                            "provenance": "expansion",
                            "source_block": tag,
                            "source_record": record["id"],
                            "expansion": occurrence,
                            "original_exchange": record.get("exchange"),
                        }
                    )
                self._assembled(model, [f"exp{occurrence}/{i}" for i in range(len(results))], results)
                operation = {
                    "version": 1,
                    "operation": "expand_block",
                    "run_id": self.run_id,
                    "tool_call_id": call_id,
                    "tag": tag,
                    "occurrence": occurrence,
                    "fingerprint": fingerprint,
                    "results": results,
                    "origins": origins,
                }
                db.execute(
                    "INSERT INTO operations VALUES (?, ?, ?, ?)", (self.run_id, call_id, occurrence, encode(operation))
                )
        except (sqlite3.Error, OSError) as exc:
            raise SelectionError(f"Expansion storage failed: {exc}. Retry after fixing local storage.") from exc
        except SelectionError:
            raise
        except (ValueError, KeyError, TypeError) as exc:
            raise SelectionError(
                f"Invalid archive for {tag}: {exc}. Repair the local archive before retrying."
            ) from exc
        self.expansions.append(operation)
        return results, origins

    def compact(
        self, action: dict, response: dict, model, format_results, *, min_chars: int, min_reduction: int
    ) -> list[dict]:
        self.projection()
        fingerprint = encode({key: action.get(key) for key in ("start_quote", "end_quote", "summary")})
        call_id = action["tool_call_id"]
        if call_id in self._applied:
            if self._applied[call_id]["fingerprint"] != fingerprint:
                raise SelectionError("Conflicting reuse of a committed compaction tool-call ID")
            return copy.deepcopy(self._applied[call_id]["results"])
        try:
            with closing(self._connect()) as db:
                if committed := self._committed(db, call_id, "compact_block", fingerprint):
                    return copy.deepcopy(committed["results"])
        except (sqlite3.Error, OSError) as exc:
            raise SelectionError(f"Archive read failed: {exc}. Retry after fixing local storage.") from exc
        if action.get("error"):
            raise SelectionError(action["error"])
        self._validate_response(response)
        selected = self._select(action)
        selected_chars = sum(
            len(text) for entry in self.trigger["entries"] if entry["id"] in selected for text in entry["visible"]
        )
        if selected_chars < min_chars:
            raise SelectionError(f"Selected range has {selected_chars} visible characters; select at least {min_chars}")
        if not isinstance(action.get("summary"), str) or not action["summary"].strip():
            raise SelectionError("Provide a nonempty agent-written summary")
        tag = f"ctx-{self.revision + 1:04d}"
        local = [key for key in selected if not (self.records[key].block or self.records[key].source_block)]
        references = [
            {
                "tag": record.block or record.source_block,
                "source_record": record.source_record or record.id,
                "selected_id": key,
                "selection_order": index,
                "expansion": record.expansion,
                "original_exchange": record.original_exchange,
            }
            for index, key in enumerate(selected)
            if (record := self.records[key]).block or record.source_block
        ]
        refs = list(dict.fromkeys(reference["tag"] for reference in references))
        summary_id = f"m{len(self.records) + 1:06d}"
        summary = model.format_message(
            role="user", content=f"{block_header(tag, refs)} {action['summary']}\n(Assistant-generated memory.)"
        )
        start = self.active.index(selected[0])
        proposed = self.active[:start] + [summary_id] + self.active[start + len(selected) :]
        summary_record = ContextMessage(
            summary_id, summary, "summary", start, content_ids(summary, summary_id), block=tag
        )
        messages = [summary if key == summary_id else self.records[key].message for key in proposed]
        self._assembled(model, proposed, messages)
        reduction = 0
        receipt_width = len(str(self.trigger["characters"]))
        for _ in range(8):
            results = format_results(
                response,
                [
                    {
                        "output": f"Compacted [{tag}]; reduced request by {reduction:{receipt_width}d} characters.",
                        "returncode": 0,
                        "exception_info": "",
                    }
                ],
            )
            measured = self.trigger["characters"] - len(encode(self._prepare(model, messages + results)))
            if measured == reduction:
                break
            reduction = measured
        else:
            raise ContextAssemblyError("Compaction receipt size failed to converge")
        if reduction < min_reduction:
            raise SelectionError(
                f"Net request reduction is {reduction} characters; need at least {min_reduction}. Shorten the summary or select more history"
            )
        event = {
            "version": 1,
            "run_id": self.run_id,
            "revision": self.revision + 1,
            "selected_ids": selected,
            "summary_id": summary_id,
            "active_before": list(self.active),
            "active_after": proposed,
            "tool_call_id": call_id,
        }
        if proposed != self._expected_active(self.events + [event]):
            raise ContextAssemblyError("Proposed projection differs from replacement history")
        if [key for key in proposed if key != summary_id] != [key for key in self.active if key not in selected]:
            raise ContextAssemblyError("Replacement changed surviving message order")
        record = {
            "version": 2,
            "run_id": self.run_id,
            "tag": tag,
            "summary": action["summary"],
            "summary_record": asdict(summary_record),
            "messages": [self.records[key].message for key in local],
            "records": [asdict(self.records[key]) for key in local],
            "hidden_ids": local,
            "selection_order": selected,
            "original_order": [self.records[key].order for key in local],
            "relationships": {key: relationships(self.records[key].message) for key in local},
            "children": refs,
            "references": references,
            "tool_call_id": call_id,
            "fingerprint": fingerprint,
            "reduction": reduction,
            "request_sequence": self.trigger["sequence"],
            "results": results,
        }
        try:
            with closing(self._connect()) as db, db:
                db.execute("BEGIN IMMEDIATE")
                self._committed(db, call_id, "compact_block", fingerprint)
                db.execute("INSERT INTO blocks VALUES (?, ?, ?, ?)", (self.run_id, call_id, tag, encode(record)))
                db.execute(
                    "INSERT INTO events VALUES (?, ?, ?, ?)", (self.run_id, self.revision + 1, call_id, encode(event))
                )
        except (sqlite3.Error, OSError) as exc:
            raise SelectionError(f"Archive write failed: {exc}. Retry after fixing local storage.") from exc
        self.records[summary_id] = summary_record
        self.events.append(event)
        self.active = proposed
        self.revision += 1
        self._applied[call_id] = record
        self.projection()
        return results

    def serialize(self) -> dict:
        return {
            "archive_path": str(self.db_path),
            "run_id": self.run_id,
            "revision": self.revision,
            "active_ids": list(self.active),
            "events": copy.deepcopy(self.events),
            "expansions": [
                {key: copy.deepcopy(value) for key, value in operation.items() if key != "results"}
                for operation in self.expansions
            ],
            "records": {
                key: {field: value for field, value in asdict(record).items() if field != "message"}
                | ({"message": copy.deepcopy(record.message)} if record.provenance == "summary" else {})
                for key, record in self.records.items()
            },
            "manifests": copy.deepcopy(self.manifests),
            "blocks": {record["tag"]: record["children"] for record in self._applied.values()},
        }
