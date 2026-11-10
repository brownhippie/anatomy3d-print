# Moving Figuro (2D to 3D) to another Windows PC

A brief for Claude Code (or a person) moving the running app, its Lab runs, studies, ledger and tuned
values from one Windows PC (the OLD PC, today the laptop at `D:\2D to 3D`) to another (the NEW PC,
the workstation). Everything runs offline on the PC itself; nothing goes to Railway or any cloud.

## What has to move, and where it lives

| What | Where on the OLD PC | Notes |
|---|---|---|
| Lab runs, studies, ledger | `<AppDir>\data\lab`, `data\study`, `data\ledger` | The work itself. ~3.5 GB, mostly `lab`. |
| Tuned values (what studies applied) | `%TEMP%\anatomy3d-tuned.json` | The first start file did not set `ANATOMY3D_TUNED`, so the app kept them in TEMP. `data\tuned.json` is the older copy from Railway. The kit takes the TEMP one and keeps the Railway one as `tuned.railway.json`. |
| AI models (downloaded once) | `%USERPROFILE%\.cache\anatomy3d-print` | Re-downloadable, but copied so the NEW PC works offline at once. |
| The app code | `<AppDir>\webapp`, `src`, … | Not copied: the NEW PC downloads branch `learning-fix` from GitHub (it has the learning fix and the autopilot status line). |
| Study render cache | `%TEMP%\anatomy3d-study-cache` | Not moved: rebuilt by itself. |

## The two scripts

- `tools/migrate/export.ps1` (OLD PC): records what is running, pauses it (autopilot off, the running
  study and Lab runs paused), waits for the build in progress to finish, stops the app, and writes
  `figuro-migration.tar` + `figuro-migration.manifest.json` (file counts, sizes, SHA-256 of every ledger
  file, what was running). It changes and deletes nothing in the data.
- `tools/migrate/import.ps1` (NEW PC): installs Python 3.12 if missing, downloads the app, installs its
  packages, unpacks the package, **checks every count, size and ledger fingerprint against the
  manifest** (and cleans up and stops if anything differs), puts the tuned values and models in place,
  writes `Start 2D to 3D.bat` (now with `ANATOMY3D_TUNED` pointing into `data`), starts the app, and
  resumes exactly what was running on the OLD PC.

Both were tested end to end (mocked Windows parts) on 2026-11-10, including a corrupted package
being refused and a clean retry after it.

## Steps

### On the OLD PC
1. Get the scripts (they are on branch `learning-fix`):
   ```powershell
   $zip = "$env:TEMP\fig.zip"; Invoke-WebRequest 'https://github.com/brownhippie/anatomy3d-print/archive/refs/heads/learning-fix.zip' -OutFile $zip -UseBasicParsing
   Expand-Archive $zip "$env:TEMP\fig" -Force; Copy-Item "$env:TEMP\fig\anatomy3d-print-learning-fix\tools\migrate\*" "$env:USERPROFILE\Desktop\" -Force
   ```
2. Export straight to where the NEW PC can read it: a USB drive (`E:\`), or a shared folder on the
   workstation (`\\WORKSTATION\share`). The app may be running; the script pauses and stops it.
   ```powershell
   powershell -ExecutionPolicy Bypass -File "$env:USERPROFILE\Desktop\export.ps1" -AppDir 'D:\2D to 3D' -OutDir 'E:\'
   ```
3. Check its last lines: `DONE. Package: … (≈3.5 GB)` and what was running.
4. **Do not start the app on the OLD PC again.** Two PCs writing would split the ledger and the held
   references (they are only ever re-learned after a real change, by design).

### Move the package
Copy `figuro-migration.tar` **and** `figuro-migration.manifest.json` to the NEW PC (USB, network share,
anything). Skip this if step 2 wrote straight to a share on the NEW PC.

### On the NEW PC
1. Put `import.ps1` somewhere (same download as step 1 above, or copy it from the OLD PC's Desktop).
2. Run it (pick the folder the app should live in; it must not already contain a `data` folder):
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\import.ps1 -Package 'E:\figuro-migration.tar' -AppDir 'D:\2D to 3D'
   ```
3. It ends with `DONE on <PC name>`, the autopilot's state, and the Lab runs with their progress. They
   must match the OLD PC's export lines (same run IDs, same done counts, the study resumed).
4. From now on start it with `Start 2D to 3D.bat` in that folder.

## Checks before telling the owner it is done
- `import.ps1` printed `Checked: every file count, size and ledger fingerprint matches the old PC.`
- `(Invoke-RestMethod http://127.0.0.1:8000/api/study).autopilot_now` shows `running` or `resting`.
- The Lab page shows run `cd40a9dbd19b` (bust_photo.png) at the same count it had on the OLD PC.
- `(Invoke-RestMethod http://127.0.0.1:8000/api/study).tuned.source` names the study the values came
  from (not empty), so the applied values came across.

## Don'ts
- Don't run the app on both PCs.
- Don't delete the OLD PC's `data` folder until the NEW PC has run for a few days. It is the only
  other copy.
- Don't use Railway or any cloud host for this: the owner wants it offline on their own PCs.
- Don't edit files inside `data` by hand.

## If something goes wrong
- `The package does not match its manifest`: the copy was damaged. Copy the `.tar` again and rerun;
  the script has already cleaned up.
- `already has a data folder`: the target folder was used before. Pick a new folder or move that
  `data` away (never delete it without asking the owner).
- `The app did not start`: read the error in the black window the start file opened.
- Python install needs `winget` (Windows 10/11 App Installer). Without it, install Python 3.12 from
  python.org (tick "Add python.exe to PATH") and rerun.
