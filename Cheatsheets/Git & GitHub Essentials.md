## 🧠 Git & GitHub Essentials — Deep Dive Notes



### 🔁 STAGING & UNSTAGING
- **Check current status**:  
  ```bash
  git status
  ```
  Shows staged, unstaged, and untracked files.



- **Stage files**:  
  ```bash
  git add <file>
  git add .       # Stage all changes
  ```



- **Unstage files**:  
  ```bash
  git reset <file>
  ```
  Removes file from staging area but keeps changes.



---



### 🌱 BRANCHING & SWITCHING
- **Create a new branch**:  
  ```bash
  git branch <branch_name>
  ```



- **Switch to a branch**:  
  ```bash
  git checkout <branch_name>
  ```



- **Create and switch in one step**:  
  ```bash
  git checkout -b <branch_name>
  ```



---



### 📜 COMMIT HISTORY & INSPECTION
- **View commit history**:  
  ```bash
  git log
  ```



- **Show details of a specific commit**:  
  ```bash
  git show <commit_hash>
  ```



- **Show only filenames changed in a commit**:  
  ```bash
  git show --name-only <commit_hash>
  ```



- **View all reference logs (including deleted branches)**:  
  ```bash
  git reflog
  ```



---



### 🔄 SYNCING & UPDATING
- **Pull latest changes from remote**:  
  ```bash
  git pull
  ```
  Combines `git fetch` + `git merge`.



---



### 🔙 UNDOING CHANGES
- **Revert a commit (safe undo)**:  
  ```bash
  git revert <commit_hash>
  ```
  Creates a new commit that undoes the changes.



- **Reset to a previous commit (3 modes)**:
  - Soft (keeps changes):  
    ```bash
    git reset --soft <commit_hash>
    ```
  - Mixed (default, unstages changes):  
    ```bash
    git reset --mixed <commit_hash>
    ```
  - Hard (removes changes):  
    ```bash
    git reset --hard <commit_hash>
    ```



---



### 🔍 COMPARING CHANGES
- **Compare branches**:  
  ```bash
  git diff <source_branch> <target_branch>
  ```



- **Compare working directory with last commit**:  
  ```bash
  git diff
  ```



---



### 🏁 INITIALIZATION & SNAPSHOTS
- **Initialize a local repo**:  
  ```bash
  git init
  ```



- **Create a tag (snapshot)**:  
  ```bash
  git tag <tag_name>
  ```



---



### 🧳 STASHING CHANGES
- **Stash current changes**:  
  ```bash
  git stash
  ```



- **Apply stashed changes**:  
  ```bash
  git stash pop
  ```



- **List all stashes**:  
  ```bash
  git stash list
  ```



---



### 🧬 REBASE & CLEANUP
- **Rebase current branch**:  
  ```bash
  git rebase <base_branch>
  ```
  Applies commits from base branch before current changes.



- **Remove untracked files**:  
  ```bash
  git clean -f
  ```
  Add `-d` to remove untracked directories.



---



### 🧭 Bonus Tips
- Use `git status` often to stay aware of your working state.
- Use `git log --oneline --graph` for a visual history.
- Use `.gitignore` to exclude files from tracking.
