---
title: VSCode基础
categories:
  - Tool
  - Editor
tags:
  - Tool
  - Editor
  - Windows
  - Linux
  - Setting
  - Keymapper
date: 2019-03-21 17:27:37
updated: 2024-07-08 15:38:17
toc: true
mathjax: true
comments: true
description: VSCode基础
---

##  Settings

-   VSCode 配置方式
    -   配置文件类型
        -   *User Settings*：用户对所有项目的配置
        -   *Workspace Settings*：针对当前项目的设置
    -   配置方式
        -   *GUI*：提示丰富
        -   *JSON*：配置快捷，`JSON` 格式

### Command Palette

-   Command Palette
    -   Python
        -   选择默认python环境
    -   Terminal
        -   选择默认终端，可能的终端包括 *CMD*、*Powershell*、*WSL*（若启用Linux子系统）

### KeyMapper

-   *KeyMapper*
    -   ``<c-s-\`>``：新建默认终端会话
    -   `<c-s-p>`：打开命令面板

### Terminal

```json
{
    "terminal.integrated.shell.windows"："/path/to/shell",
        // 默认shell
    "terminal.integrated.shellArgs.windows": [ ],
    "terminal.integrated.shellArgs.linux": [ ],
    "terminal.integrated.shellArgs.osx": [ ],
        // VSCode创建terminal启动参数（3系统分别配置）
}
```

-   VSCode 终端
    -   支持 integrated 内置集成、external 外置
    -   兼容所有大部分终端 Shell，如
        -   `C:/Windows/System32/cmd.exe`
        -   `C:/Windows/System32/powershell.exe`
        -   `C:/Windows/System32/wsl.exe`
        -   `/path/to/git/bin/bash.exe`：Git 自带 *MSYS* Bash 等

### Python

```json
{
    "python.condaPath": "/path/to/conda/Scripts",
        // conda安装目录Scripts文件夹
    "python.venvPath": "/path/to/conda/envs",
        // 虚拟环境目录，VSCode会在其中查找虚拟环境，作为
        // Command Palette中的备选项
    "python.pythonPath": "/path/to/python.exe",
        // 默认python解释器路径
    "python.terminal.activateEnvironment": true,
        // 创建python shell时，尝试中激活虚拟环境
}
```

-   Python 环境
    -   `python.pythonPath` 默认 Python 环境
        -   `python.terminal.activateEnviroment` 置位时，VSCode 会尝试执行 `python.pythonPath` 同级中`Scripts/activate.bat`激活虚拟环境
        -   虚拟环境需要安装 `conda`，否则没有无法正常激活默认虚拟环境

### CPPC

####    配置文件

-   `.vscode/c_cpp_properties.json`

    ```json
    {
        "configurations": [
            {
                "name": "WSL",
                "includePath": [
                    "${workspaceFolder}/**"
                ],
                "defines": [
                    "LOCAL",
                    "_DEBUG",
                    "UNICODE",
                    "_UNICODE"
                ],
                "compilerPath": "/usr/bin/gcc",
                "cStandard": "c11",
                "cppStandard": "c++14",
                "intelliSenseMode": "gcc-x64"
            }
        ],
        "version": 4
    }
    ```

    -   `C/C++`项目基本配置

-   `.vscode/tasks.json`：利用 VSCode 的 *Tasks* 功能调用 WSL 的 *GCC/G++* 编译器

    ```json
    {
        "version": "2.0.0",
        "tasks": [
            {
                "label": "Build",
                "command": "g++",
                "args": [
                    "-g",
                    "-Wall",
                    "-std=c++14",
                    "/path/to/${fileBasename}",
                    "-o",
                    "/path/to/a.out",
                    "-D",
                    "LOCAL"
                ],
                "problemMatcher": {
                    "owner": "cpp",
                    "fileLocation": [
                        "relative",
                        "${workspaceRoot}"
                    ],
                    "pattern": {
                        "regexp": "^(.*):(\\d+):(\\d+):\\s+(warining|error):\\s+(.*)$",
                        "file": 1,
                        "line": 2,
                        "column": 3,
                        "severity": 4,
                        "message": 5
                    }
                },
                "type": "shell",
                "group": {
                    "kind": "build",
                    "isDefault": true
                },
                "presentation": {
                    "echo": true,
                    "reveal": "silent",
                    "focus": true,
                    "panel": "shared"
                }
            },
            {
                "label": "Run",
                "command": "/path/to/a.out",
                "type": "shell",
                "dependsOn": "Build",
                "group": {
                    "kind": "test",
                    "isDefault": true
                },
                "presentation":{
                    "echo": true,
                    "reveal": "always",
                    "focus": true,
                    "panel": "shared",
                    "showReuseMessage": true
                }
            }
        ]
    }
    ```

-   `.vscode/launch.json`：*gdb* 调试配置

    ```json
    {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "(gdb) Bash on Windows Launch",
                "type": "cppdbg",
                "request": "launch",
                "program": "/path/to/a.out",
                "args": ["-f", "Threading"],
                "stopAtEntry": false,
                "cwd": "/path/to/cppc/",
                "environment": [],
                "externalConsole": true,
                "MIMode": "gdb",
                "pipeTransport": {
                    "debuggerPath": "/usr/bin/gdb",
                    "pipeProgram": "C:\\windows\\system32\\bash.exe",
                    "pipeArgs": ["-c"],
                    "pipeCwd": ""
                },
                "setupCommands": [
                    {
                        "description": "Enable pretty-printing for gdb",
                        "text": "-enable-pretty-printing",
                        "ignoreFailures": false
                    }
                ],
                "sourceFileMap": {
                    "/mnt/c": "c:\\",
                    "/mnt/d": "d:\\"
                },
                "preLaunchTask": "Build"
            },
        ]
    }
    ```

### Git

```json
{
    "git.ignore.MissingGitWarning": true,
    "git.path": "/path/to/xxxgit.exe"
}
```

-   Git 配置
    -   `git.path` 除可为一般 `git.exe` 外，也可以是 “伪装” Git
        -   工具[wslgit](https://github.com/andy-5/wslgit) 可让 VSCode 直接使用 WSL 内的 Git

