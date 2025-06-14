## 目录架构

./docs 中定义了网站主体结构，引用style.css样式文件（一般不要修改样式文件）。

./docs/testlab 中定义了课程内容，包含课程和实验内容。
./docs/index.md 是网站的首页。
./docs/testlab/Lesson 1.md 是课程内容的示例，包含课程介绍、实验内容、实验文件等。
./docs/testlab/Lesson 1_Experiment.md 是实验内容的示例，包含实验目的、实验步骤、实验结果等。

增加课程章节内容请在此目录下增加，并修改docs/index.md 中的课程目录。


## 本地测试
使用以下命令在本地测试网站：

```bash
pip install mkdocs-material
mkdocs serve
```

## 部署
将整个项目git push上传到GitHub的main分支后，使用GitHub Pages功能部署网站。（已配置github pages，无需修改）。

还需要在本地使用以下命令生成部署分支gh-pages，**只有在生成gh-pages分支后，修改才能反映到github pages**：

```bash
mkdocs gh-deploy
```

然后打开课程主页测试。课程主页链接：https://wzbxpy.github.io/LLM-System-and-Engineering/
