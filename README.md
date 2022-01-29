# [CellSeg](http://dx.doi.org/10.1186/s12859-022-04570-9)

Fork for Docker

```
docker pull andrewrech/cellseg:latest
```

```
docker run -it --rm \
  -v /toplevel/data/folder:/home/jovyan/work \
  -e GRANT_SUDO=yes \
  andrewrech/cellseg:latest
```

```
source ~/CellSeg/start.bash
```
