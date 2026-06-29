### Problem

Folders appear with a **green background** in the WSL terminal because the `LS_COLORS` variable was modified.

<img width="868" height="76" alt="50dc0674-e6a5-4ca0-ad51-6458b64a5139" src="https://github.com/user-attachments/assets/090dd60c-f66d-41bc-b66d-dc2d8f44a764" />

### Solution

Add this to end of the .bashrc file

```
LS_COLORS=$LS_COLORS:'ow=1;34:' ; export LS_COLORS
```
