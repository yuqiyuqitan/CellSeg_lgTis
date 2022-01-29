from cvconfig import CVConfig
from main import main

# paths to process
targets = [
    "/home/jovyan/work/",
]

for target in targets:
    cf = CVConfig(target, 1)
    print(cf.DIRECTORY_PATH)
    main(cf)
