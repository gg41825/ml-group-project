### Step 1.
#### Go to the desired path on your local environment and pull this repository using the following command.

```
git clone {repository_path}
```

### Step 2.
#### Create a new branch, we will work on our own branch.

```
git checkout -b {your_branch_name}
```

##### ex.

```
git clone https://github.com/gg41825/ml-group-project.git
```
```
git checkout -b ginny
``` 

### Step 3.
#### If you have finished your parts and want to push them to remote repository:
```
git add {file}
```
### Step 4.
#### Write some commit message.
```
git commit -m {your_commit_message}
```
### Step 5.
#### Push the commit to remote repository
```
git push        ## If it is the first time that you push files to your remote branch, git will ask you to set upstream, just follow the warning hint.
```

##### ex.

You have files
ooxx.py

```
git add ooxx.py
```
```
git commit -m 'my first commit'
```
```
git push
```