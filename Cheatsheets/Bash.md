### Problem

Folders appear with a **green background** in the WSL terminal because the `LS_COLORS` variable was modified.

<img width="868" height="76" alt="50dc0674-e6a5-4ca0-ad51-6458b64a5139" src="https://github.com/user-attachments/assets/090dd60c-f66d-41bc-b66d-dc2d8f44a764" />

### Solution

Reset the directory color to the default **bold blue**:

```bash
export LS_COLORS='di=01;34'
```

**Explanation:**

* `LS_COLORS` → Controls colors for `ls --color`.
* `di` → Directory color.
* `01` → Bold.
* `34` → Blue.

**Result:** Directories are displayed in **bold blue** instead of a green background.
