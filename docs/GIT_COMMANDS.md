# Git常用命令
## 查看文件状态：git status
- 查看文件修改：git diff
- 查看提交历史：git log
- 查看当前版本信息：git branch -a
- 查看文件大小: 
    git ls-files -z --others --cached --exclude-standard <path> | xargs -0 du -ch | tail -n1
    git ls-files -z --others --cached --exclude-standard | xargs -0 du -ch | tail -n1

## 设置别名
- 一键提交改动: git config --global alias.lazy "!git add . && git commit -m 'update' && git push"
    git lazy = git add . -> git commit -m 'update' -> git push

## 分支管理
- 查看分支：git branch
- 创建分支：git branch <branch-name>
- 切换分支：git checkout <branch-name>
- 创建并切换分支：git checkout -b <branch-name>
- 删除分支：git branch -d <branch-name>
- 推送分支：git push origin <branch-name>
- 拉取分支：git pull origin <branch-name>
- 合并分支：git merge <branch-name>
- 查看远程分支：git branch -r
- 查看所有分支：git branch -a

## 子模块管理
- 添加子模块：git submodule add <url> <path>
- 设置子模块分支：git submodule set-branch --branch <branch-name> <path>
- 拉取子模块分支内容并更新: git submodule update --remote
- 提交改动到主仓库: git add .gitmodules | git commit -m "update submodule" | git push


- 初始化子模块：git submodule init
- 更新子模块：git submodule update
- 删除子模块：git submodule deinit <path>
- 查看子模块：git submodule status
- 查看子模块详细信息：git submodule status --recursive

## Access Token
- 永久记住(有风险): git config --global credential.helper store
