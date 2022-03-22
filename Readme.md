# For requirements.txt
Try
```
python=pip list --format=freeze > requirements.txt
```
and rerun. It looks like this is caused by changing the behavior of
```
pip freeze
```

# packages for current folder (Suggested for  create requirments.txt)
```
    pip install pipreqs
```
```
    pipreqs --encoding=UTF-8 --force
```