# 7 - Dify 的 Windows 平台部署

本章偏**实操部署**：在 Windows 上把 Dify 跑在本地，实现**私有化**使用 Dify 开发 Agent、工作流和 RAG 应用。流程比 [第 6 章 Coze 部署](6-Coze的Windows平台部署.md) 更简单，核心就是「装 Docker → 拿代码 → 改配置 → 一条命令启动」。

---

**本章课程目标：**

- 在 Windows 上正确安装 **Docker Desktop**，并能用命令行 `docker` 验证安装成功；遇到 WSL 提示时选择安装。
- 获取 **Dify** 源码（GitHub 或网盘），进入 **docker** 目录，将 `.env.example` 复制为 `.env` 并按需修改（如端口改为 8100 等未被占用的端口）。
- 在 docker 目录下执行 `docker compose up -d` 完成部署，能在浏览器访问 Dify（如 `http://localhost` 或 `http://localhost:8100`），并完成首次用户名/密码设置。
- 知道若镜像拉取失败可多试几次或检查网络，使用方式与 Dify 云平台一致。

**前置知识建议：** 已用过 Dify 云端或在 [第 3 章](3-基于Coze&Dify平台的智能体开发.md) 中接触过 Dify 会更易理解本地版用途；会基本命令行（打开 cmd、进入目录）即可。若未接触过 Docker，按第 1 节步骤安装即可。

**入门阅读提示：** 先完成第 1 节 Docker 安装并重启（必要时安装 WSL），再用第 2 节：获取 Dify → 进入 docker 目录 → 复制并修改 .env（端口冲突时必改）→ 执行 `docker compose up -d`。首次拉镜像可能较慢，报错可重试。官方文档见：https://docs.dify.ai/zh/self-host/quick-start/docker-compose 。

---

## 1. Docker Desktop 安装

### 1.1 下载安装包

官网：https://www.docker.com/

方式 1：选择版本下载：

![](images/7/7-1-1-1.png)

方式 2：从网盘资料里获取：

![](images/7/7-1-1-2.png)

### 1.2 点击 OK 即可

![](images/7/7-1-2-1.png)

### 1.3 等待安装完成

![](images/7/7-1-3-1.png)

### 1.4 重启电脑

![](images/7/7-1-4-1.png)

也可以关闭窗口，稍后自行重启。

### 1.5 接受服务协议

**重启后自动弹出**

![](images/7/7-1-5-1.png)

### 1.6 完成安装

![](images/7/7-1-6-1.png)

### 1.7 允许控制

![](images/7/7-1-7-1.png)

### 1.8 不登录使用

![](images/7/7-1-8-1.png)

### 1.9 跳过

![](images/7/7-1-9-1.png)

> 个别首次安装时，Windows 会提示需安装「适用于 Linux 的 Windows 子系统」（WSL）。选择确认安装，稍等片刻即可完成。

### 1.10 安装成功验证

按 **Win+R** 打开「运行」对话框，输入 `cmd` 并回车。

![](images/7/7-1-10-1.png)

在命令行中输入 `docker` 并回车，若出现如下内容即表示安装成功。

![](images/7/7-1-10-2.png)

## 2. 部署 Dify

### 2.1 拉取 Dify 代码

**GitHub 地址**：

> https://github.com/langgenius/Dify

或者从网盘资料中获取：

![](images/7/7-2-1-1.png)

### 2.2 修改配置

> 下面的操作，在 Dify 官方文档中有。
>
> https://docs.dify.ai/zh/self-host/quick-start/docker-compose
>
> 这里直接操作：

进入 Dify 仓库下的 **docker** 目录，将 `.env.example` 复制并重命名为 `.env`。

![](images/7/7-2-2-1.png)

然后按需修改 `.env` 中的配置即可。

> 例如：默认端口为 80，可改为 8100 等未被占用的端口。

![](images/7/7-2-2-2.png)

> **可这样记：** 部署 Dify 就两件事：**环境**（装好 Docker）和**启动**（拿代码 → 进 docker 目录 → 复制 .env 并改端口 → `docker compose up -d`）。端口 80 被占用时一定要改 .env 里的端口，否则无法访问。

### 2.3 打开终端

以下操作可在 **Windows 命令行**（cmd）或 **Docker Desktop 自带终端**中进行。

例如：在终端中进入 Dify 仓库下的 **docker** 目录。

![](images/7/7-2-3-1.png)

### 2.4 安装

执行以下命令部署 Dify：

> ```shell
> docker compose up -d
> ```

> **注意**：若因网络或镜像拉取失败报错，可多试几次或稍后重新执行该命令。

![](images/7/7-2-4-1.png)

安装完成后：

![](images/7/7-2-4-2.png)

再次执行 `docker compose up -d` 即可启动。

![](images/7/7-2-4-3.png)

在 Docker Desktop 的 **Containers（容器）** 中可查看当前运行的容器。

![](images/7/7-2-5-3.png)

### 2.5 访问 Dify

在浏览器中访问 **http://localhost**（若已修改端口为 8100，则访问 http://localhost:8100）即可。

首次访问需设置用户名与密码。

![](images/7/7-2-5-2.png)

使用方式与 Dify 官方云平台一致。

![](images/7/7-2-5-1.png)
