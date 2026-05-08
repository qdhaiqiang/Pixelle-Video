# Bilibili 视频上传使用指南

Pixelle-Video 支持在视频生成完成后自动上传到 Bilibili（B站）。本指南介绍远程用户如何配置自己的 Bilibili 账号进行上传。

---

## 工作原理

由于 Bilibili 没有提供网页上传 API，我们使用 [biliup-rs](https://github.com/biliup/biliup-rs) 命令行工具进行上传。

**上传流程**：
1. 用户在本地电脑登录 B 站，生成 `cookies.json` 文件
2. 将 `cookies.json` 上传到 Pixelle-Video 服务器
3. 视频生成完成后，服务器使用该 cookie 自动上传

---

## 步骤 1：安装 biliup-rs（本地电脑）

### macOS
```bash
# 方式 1：使用 Homebrew
brew tap biliup/biliup
brew install biliup

# 方式 2：下载预编译二进制
curl -LO https://github.com/biliup/biliup-rs/releases/download/v0.2.4/biliupR-v0.2.4-x86_64-macos.tar.xz
tar xf biliupR-v0.2.4-x86_64-macos.tar.xz
sudo mv biliupR-v0.2.4-x86_64-macos/biliup /usr/local/bin/
```

> M1/M2/M3 Mac 用户请下载 `aarch64-macos` 版本

### Linux
```bash
curl -LO https://github.com/biliup/biliup-rs/releases/download/v0.2.4/biliupR-v0.2.4-x86_64-linux.tar.xz
tar xf biliupR-v0.2.4-x86_64-linux.tar.xz
sudo mv biliupR-v0.2.4-x86_64-linux/biliup /usr/local/bin/
```

### Windows
1. 下载：https://github.com/biliup/biliup-rs/releases/download/v0.2.4/biliupR-v0.2.4-x86_64-windows.zip
2. 解压后将 `biliup.exe` 放到任意目录
3. 将该目录添加到系统 PATH

---

## 步骤 2：登录 B 站并生成 Cookie

在本地电脑的终端运行：

```bash
biliup -u ./cookies.json login
```

按提示完成登录：
1. 选择登录方式（短信/密码）
2. 输入手机号和国家代码（如 86）
3. 输入验证码
4. 登录成功后，当前目录会生成 `cookies.json` 文件

> ⚠️ 请勿分享 `cookies.json` 文件，它包含您的登录凭证。

---

## 步骤 3：在 Pixelle-Video 中上传 Cookie

1. 打开 Pixelle-Video，进入「影视解说」页面
2. 在「📺 Bilibili 上传设置」区域勾选「生成完成后自动上传至 Bilibili」
3. 在「步骤 2：上传 Cookie 文件」区域，点击上传按钮选择刚才生成的 `cookies.json`
4. 填写视频标题、标签、分区等信息（可选）
5. 正常生成视频，完成后会自动上传到您的 B 站账号

---

## 注意事项

### 多用户使用
- 每个用户的 cookie 文件是独立的，上传到服务器后相互隔离
- 服务器端按文件名区分不同用户的 cookie
- 用户可以替换上传新的 cookie 文件

### Cookie 有效期
- Bilibili cookie 通常有效期较长（数月）
- 如果上传失败提示"登录过期"，需要重新执行步骤 2 生成新的 cookie

### 服务器本机用户
- 如果您就在服务器上操作，可以直接在输入框中填写本地 cookie 路径（如 `~/cookies.json`）
- 无需通过文件上传器上传

---

## 故障排查

| 问题 | 解决方案 |
|---|---|
| "Cookie file not found" | 确认已上传 cookies.json 或路径填写正确 |
| "Bilibili login failed" | cookie 可能已过期，重新运行 `biliup login` |
| "biliup command not found" | 服务器自动安装失败，请手动安装 biliup-rs |
| 上传速度慢 | 在系统配置中切换上传线路（bda2/ws/qn/tx）|
