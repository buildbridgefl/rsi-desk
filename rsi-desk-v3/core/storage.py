"""
CSV storage that survives Streamlit Cloud restarts.

GitHub mode (token + repo set): reads/writes one CSV file in a repo through
the GitHub contents API. Use a PRIVATE repo so your trades stay private.

Local mode: a plain CSV on disk. Fine on your own computer, but Streamlit
Cloud wipes it whenever the app sleeps.
"""

import base64
import io
import os

import pandas as pd
import requests

API = "https://api.github.com"


class CSVStore:
    def __init__(self, columns, path, token=None, repo=None, branch="main"):
        self.columns = list(columns)
        self.path = path
        self.token, self.repo, self.branch = token, repo, branch
        self.remote = bool(token and repo)
        self._sha = None  # GitHub needs the current file version to overwrite it

    @property
    def label(self) -> str:
        if self.remote:
            return f"GitHub · {self.repo}/{self.path}"
        return f"local file · {self.path}"

    # ------------------------------------------------------------ helpers
    def _headers(self):
        return {"Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28"}

    def _url(self):
        return f"{API}/repos/{self.repo}/contents/{self.path}"

    def _conform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c in self.columns:
            if c not in df.columns:
                df[c] = pd.NA
        extra = [c for c in df.columns if c not in self.columns]
        return df[self.columns + extra].reset_index(drop=True)

    # --------------------------------------------------------------- API
    def load(self) -> pd.DataFrame:
        if not self.remote:
            if os.path.exists(self.path) and os.path.getsize(self.path):
                return self._conform(pd.read_csv(self.path))
            return self._conform(pd.DataFrame())

        r = requests.get(self._url(), headers=self._headers(),
                         params={"ref": self.branch}, timeout=15)
        if r.status_code == 404:
            self._sha = None
            return self._conform(pd.DataFrame())
        r.raise_for_status()
        body = r.json()
        self._sha = body["sha"]
        raw = base64.b64decode(body["content"]).decode()
        if not raw.strip():
            return self._conform(pd.DataFrame())
        return self._conform(pd.read_csv(io.StringIO(raw)))

    def save(self, df: pd.DataFrame, message: str = "update trades") -> pd.DataFrame:
        df = self._conform(df)
        text = df.to_csv(index=False)

        if not self.remote:
            folder = os.path.dirname(os.path.abspath(self.path))
            os.makedirs(folder, exist_ok=True)
            with open(self.path, "w") as f:
                f.write(text)
            return df

        payload = {"message": message, "branch": self.branch,
                   "content": base64.b64encode(text.encode()).decode()}
        if self._sha:
            payload["sha"] = self._sha
        r = requests.put(self._url(), headers=self._headers(),
                         json=payload, timeout=15)
        if r.status_code in (409, 422):
            raise RuntimeError("the trades file changed somewhere else — "
                               "press Reload, then try again")
        if r.status_code in (401, 403):
            raise RuntimeError("GitHub refused the save — check that the token "
                               "has Contents: Read and write on this repo")
        r.raise_for_status()
        self._sha = r.json()["content"]["sha"]
        return df
