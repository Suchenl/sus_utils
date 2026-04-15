# Common Commands

## Download from HuggingFace
### If too slow
- use mirror: export HF_ENDPOINT=https://hf-mirror.com

### template
huggingface-cli download <model_name> --local-dir <local_dir> --local-dir-use-symlinks False --resume-download

### use
huggingface-cli download ali-vilab/VACE-Annotators --local-dir /m2v_intern/chenyuzhuo03/MODELS/VideoGen/VACE/models --local-dir-use-symlinks False --resume-download

## Linux 文件管理

### 查看文件数量

- 统计当前目录及其子目录中的所有文件： find ./ -type f | wc -l -type f 表示只统计普通文件。

- 仅统计当前目录下的文件（不包括子目录）： find ./ -maxdepth 1 -type f | wc -l 
    `-maxdepth 1 限制搜索深度为当前目录。

- 查看特定路径下 .mp4 文件的数量
    find /path/to/search -name "*.mp4" -type f | wc -l
    
