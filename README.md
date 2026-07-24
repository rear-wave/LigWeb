# LigWeb

LigWeb 是独立的局域网 `.lig` 波形审核、分类和训练服务。浏览器端无需安装软件；服务器负责读取数据、ONNX 推理、人工纠错、纠错模型训练和主模型夜间训练。

## 数据目录

数据始终位于宿主机，不进入仓库：

- `Desktop\train_data`：五分类训练集。
- `Desktop\correct_data`：只保存五个类别目录中的已审核纠错数据。
- `LigWeb\runtime`：反馈数据库、待处理文件、导出、主/纠错模型、
  训练输出、备份和调度状态。

路径可通过 `.env.example` 中的 `LIGWEB_*` 环境变量覆盖。
`runtime/` 已加入 Git 忽略列表，并作为持久卷挂载到 Docker 容器。

加入纠错集时会逐个波形片段采用“人工结果优先，否则采用纠错模型结果”，
再分别写入 `IC/NCG/NNBE/PCG/PNBE` 目录。一个来源 `.lig` 中的不同类别
不会再被整体放进同一类别目录。

## IC 自动同步

LigWeb 每 60 秒检查一次 `correct_data\IC`，并在完成一个 LIG 审核后立即同步。仅将训练集中不存在的波形写入：

```text
train_data\IC\_ligweb_sync\
```

同步按波形哈希去重；已经手工放入训练集的 IC 会被跳过。若 IC 后续被重新纠正为其他类别，LigWeb 会移除对应的托管副本。每天 22:00 的主模型任务会先强制同步，再训练五类别分类；纠错目录中的 IC 不会被重复加入训练视图。

手动核对同步状态：

```powershell
python -m tools.sync_ic_data
```

## 启动

推荐使用 Docker：

```powershell
cd C:\Users\Administrator\Desktop\LigWeb
.\run_web_docker.bat
```

本机访问 `http://127.0.0.1:8088`，局域网访问 `http://<服务器IP>:8088`。

本机 Python 运行：

```powershell
pip install -r requirements.txt
.\run_web.bat
```

服务必须保持一个 Uvicorn worker，避免重复加载 ONNX 会话和重复启动训练任务。

## 训练自动化

- 纠错模型：北京时间每逢整点检查，有新增反馈才训练。
- 纠错模型只把“确实改变主模型类别”的结果计入验证精度；至少需要两个
  同类错分邻居，并通过相似度、正确样本边界和冲突类别三重检查才会自动改类。
- 纠错模型与主模型哈希绑定；主模型更新后会自动训练兼容的新纠错代次，
  不会继续加载旧主模型的纠错权重。
- 主模型：每天 22:00 使用 `conda` 的 `ligclassify` 环境训练。
- 主模型当前只做 IC、NCG、NNBE、PCG、PNBE 分类，不使用距离损失或距离指标。
- 新模型通过 ONNX 一致性校验后才会替换现用模型；失败时保留旧模型。

## 开发验证

```powershell
pip install -r requirements-dev.txt
python -m compileall -q ligweb tools
python -m pytest -q
```

FastAPI 入口为 `ligweb/app.py`，业务逻辑位于 `ligweb/service.py`，静态前端位于 `ligweb/static/`。
