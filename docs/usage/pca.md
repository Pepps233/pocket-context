# `pca`

`pca` runs Pocket Context Agent in your current working directory.
Use it to inspect a repository, edit files, and run commands with the configured model.

## Start a task

Choose one of these commands to run in your terminal:

```bash
pca
pca --task "Explain how this repository works"
pca --model "provider/model-name" --task "Describe the architecture"
pca --help
```

When `pca` asks **What do you want to do?**, type only the task description.
Press **Esc, then Enter** to submit it.
Enter by itself inserts a new line in this multiline prompt.
The `pca --task ...` form bypasses this prompt.

## Command line options

`pca --help` lists the options supported by your installed version.
The short option `-h` is not supported.

| Option | Purpose |
| --- | --- |
| `--help` | Show command line help and exit. |
| `-t`, `--task TEXT` | Supply the task instead of entering it interactively. |
| `-m`, `--model NAME` | Select the model; otherwise use the configured model or `MSWEA_MODEL_NAME`. |
| `-y`, `--yolo` | Execute proposed actions without confirmation. |
| `-l`, `--cost-limit AMOUNT` | Set the model cost limit in USD; `0` disables this limit. |
| `-c`, `--config SPEC` | Load a configuration or override; repeat to merge multiple specifications in order. |
| `-o`, `--output PATH` | Save the run trajectory to this file. |
| `--model-class CLASS` | Select a model interface by registered name or import path. |
| `--agent-class CLASS` | Select an agent class by registered name or import path. |
| `--environment-class CLASS` | Select an execution environment by registered name or import path. |
| `--exit-immediately` | Exit when the agent finishes without prompting for another task. |
| `--install-completion` | Install shell completion for the current shell. |
| `--show-completion` | Print shell completion for the current shell. |

The default configuration is `mini.yaml`, or the file selected by `MSWEA_MINI_CONFIG_PATH`.
The configuration filename and environment variable retain their existing names.
Supplying `--config` replaces the default configuration list, so include `-c mini.yaml` when adding overrides:

```bash
pca -c mini.yaml -c agent.cost_limit=2 --task "Explain the test suite"
pca --output ./trajectory.json --exit-immediately --task "Describe this project"
```

See [global configuration](../advanced/global_configuration.md) for model settings, API keys, and configuration paths.
Extra utilities retain their separate command, `mini-extra`; run `mini-extra --help` or `mini-extra config setup`.

## Interactive commands and keyboard shortcuts

These slash commands are handled at agent interaction prompts, such as action confirmation prompts.
The initial multiline task prompt accepts task text instead of slash commands.

| Command or key | Effect |
| --- | --- |
| `/h` | Show the current mode and available interactive commands. |
| `/c` | Switch to confirmation mode. |
| `/y` | Switch to automatic execution mode and approve the pending actions at a confirmation prompt. |
| `/u` | Switch to human mode; at a confirmation prompt, skip the proposed actions. |
| `/m` | Enter a multiline comment; submit it with Esc, then Enter. |
| Enter at an action confirmation prompt | Approve the pending actions. |
| A comment at an action confirmation prompt | Reject the pending actions and send your feedback to the agent. |
| Esc, then Enter in a multiline prompt | Submit the task or comment. |
| Up/Down in a multiline prompt | Navigate input history. |
| Ctrl+R in a multiline prompt | Search input history. |
| Ctrl+C during a run | Interrupt the agent and return to an interaction prompt. |
| Enter when the agent asks for another task | Finish the session. |

The default mode is `confirm`.
In confirmation mode, review proposed commands before pressing Enter to execute them.
In `yolo` mode, commands execute automatically.
In `human` mode, you enter the commands to execute.
Use `--yolo` to start in automatic execution mode.

The default trajectory file is `last_mini_run.traj.json` in the global configuration directory.
The startup output prints the configuration path.
Use `--output` to choose a different trajectory path.

## Implementation

??? note "Default config"

    - [Read on GitHub](https://github.com/Pepps233/pocket-context/blob/main/pocketcontextagent/config/mini.yaml)

    ```yaml
    --8<-- "pocketcontextagent/config/mini.yaml"
    ```

??? note "Run script"

    - [Read on GitHub](https://github.com/Pepps233/pocket-context/blob/main/pocketcontextagent/run/mini.py)
    - [API reference](../reference/run/pca.md)

    ```python
    --8<-- "pocketcontextagent/run/mini.py"
    ```

??? note "Agent class"

    - [Read on GitHub](https://github.com/Pepps233/pocket-context/blob/main/pocketcontextagent/agents/interactive.py)
    - [API reference](../reference/agents/interactive.md)

    ```python
    --8<-- "pocketcontextagent/agents/interactive.py"
    ```

{% include-markdown "../_footer.md" %}
