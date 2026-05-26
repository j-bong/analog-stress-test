# First-time GitHub push

For someone setting up GitHub for the first time.

## 1. Make a GitHub account

If you don't have one, sign up at https://github.com .

## 2. Install Git and the GitHub CLI

Windows:
- Git: install Git for Windows from https://git-scm.com (it includes Git Bash).
- GitHub CLI: install with `winget install --id GitHub.cli`
  (or download from https://cli.github.com).

macOS:
```
brew install git gh
```

Linux:
```
sudo apt install git
sudo apt install gh        # or: see https://cli.github.com
```

## 3. Authenticate

Open Git Bash (Windows) or any terminal, then:

```
gh auth login
```

Pick:
- GitHub.com
- HTTPS
- Login with a web browser  (the CLI shows you a code, paste it in the browser)

When `gh auth status` prints your username, you're set.

## 4. Push this repo

Unzip the bundle, open a terminal in the unzipped folder, and run:

```
bash push_to_github.sh
```

The script:
- creates the repo on GitHub (public, no description, no topics, no issues, no wiki)
- pushes everything in one commit
- prints the URL to share

If you want a different repo name:

```
bash push_to_github.sh my-repo-name
```

## 5. Keep it low-profile

After the push, on github.com:
- do NOT click "Pin" on the repo (Pinned repos appear at the top of your profile)
- do NOT add a description or topics
- do NOT enable GitHub Pages

The repo will still be listed on your profile under the "Repositories" tab
(GitHub has no true unlisted mode), but it won't surface in search results
without those metadata.

## 6. Replace / update later

```
git add -A
git commit -m "update"
git push
```
