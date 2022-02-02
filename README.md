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

Edit `run.py` to point to your data folders, then run:
```
source ~/CellSeg/start.bash
```

```
...
Segmented crop in 5.329453229904175 seconds.
Segmented crop in 3.884769916534424 seconds.
Segmented crop in 4.497356414794922 seconds.
Segmented crop in 4.965950965881348 seconds.
...
```
