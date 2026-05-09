# Copyright (C) 2025 AIDC-AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Bilibili Uploader Service

Wraps biliup-rs CLI for uploading videos to Bilibili.
Requires biliup binary in PATH and a valid cookie file.
"""

import json
import re
import subprocess
import shlex
import urllib.request
from pathlib import Path
from typing import List, Optional, Dict

from loguru import logger

from pixelle_video.services.biliup_installer import ensure_biliup


class BilibiliUploader:
    """Bilibili video uploader using biliup-rs."""

    def __init__(self, cookie_path: str, biliup_cmd: Optional[str] = None):
        """
        Initialize uploader.

        Args:
            cookie_path: Path to biliup cookie file (e.g. cookies.json)
            biliup_cmd: biliup binary name or path (auto-detected if None)
        """
        self.cookie_path = str(Path(cookie_path).expanduser().resolve())
        # Auto-install biliup if not found
        if biliup_cmd is None:
            self.biliup_cmd = ensure_biliup()
        else:
            self.biliup_cmd = biliup_cmd

    def _run(self, cmd: List[str], timeout: int = 600) -> subprocess.CompletedProcess:
        """Run biliup command."""
        logger.debug(f"[biliup] {' '.join(shlex.quote(c) for c in cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.stdout:
            logger.debug(f"[biliup stdout] {result.stdout}")
        if result.stderr:
            logger.debug(f"[biliup stderr] {result.stderr}")
        return result

    def check_login(self) -> bool:
        """Check if login is valid by running a dry upload --help."""
        try:
            result = self._run(
                [self.biliup_cmd, "-u", self.cookie_path, "upload", "--help"],
                timeout=10
            )
            return result.returncode == 0
        except FileNotFoundError:
            logger.error("biliup command not found. Please install biliup-rs.")
            return False
        except Exception as e:
            logger.error(f"Login check failed: {e}")
            return False

    def upload(
        self,
        video_path: str,
        title: Optional[str] = None,
        desc: Optional[str] = None,
        tags: Optional[List[str]] = None,
        extra_tags: str = "",
        tid: int = 228,
        copyright: int = 1,
        cover: Optional[str] = None,
        dynamic: str = "",
        dtime: Optional[int] = None,
        line: str = "bda2",
        limit: int = 3,
        collection_id: Optional[int] = None,
    ) -> str:
        """
        Upload a video to Bilibili.

        Args:
            video_path: Path to video file
            title: Video title (default: filename stem)
            desc: Video description
            tags: List of tags
            extra_tags: Comma-separated extra tags string
            tid: Zone TID (228=电影, 230=电视剧, etc.)
            copyright: 1=自制, 2=转载
            cover: Cover image path
            dynamic: Space dynamic text
            dtime: Scheduled publish timestamp
            line: Upload line
            limit: Concurrent upload limit

        Returns:
            bvid string (e.g. "BV1xx411c7mD")

        Raises:
            RuntimeError: If upload fails
        """
        vp = Path(video_path)
        if not vp.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        if not Path(self.cookie_path).exists():
            raise FileNotFoundError(
                f"Cookie file not found: {self.cookie_path}. "
                f"Please run: biliup -u {self.cookie_path} login"
            )

        # Build title
        upload_title = title or vp.stem
        # Bilibili title max 80 chars
        # Preserve segment suffix (e.g. "（第 X 段）" / "(Segment X)") when truncating
        segment_match = re.search(r'[（(][^）)]*(?:第|Segment)\s*\d+[^）)]*[）)]$', upload_title)
        if segment_match:
            suffix = segment_match.group()
            base = upload_title[:segment_match.start()]
            max_base = 80 - len(suffix)
            upload_title = base[:max_base] + suffix
        else:
            upload_title = upload_title[:80]

        # Build tags
        all_tags: List[str] = []
        if tags:
            all_tags.extend([t.strip() for t in tags if t.strip()])
        if extra_tags:
            all_tags.extend([t.strip() for t in extra_tags.split(",") if t.strip()])
        # Deduplicate and limit to 12
        seen = set()
        deduped = []
        for t in all_tags:
            if t not in seen and len(t) <= 20:
                seen.add(t)
                deduped.append(t)
        all_tags = deduped[:12]
        if not all_tags:
            all_tags = ["影视解说", "原创"]

        # Build description
        upload_desc = desc or f"{upload_title}\n\n自动上传于 Pixelle-Video"

        # Build command
        cmd = [
            self.biliup_cmd,
            "-u", self.cookie_path,
            "upload", str(vp),
            "--title", upload_title,
            "--desc", upload_desc,
            "--tag", ",".join(all_tags),
            "--tid", str(tid),
            "--copyright", str(copyright),
            "--line", line,
            "--limit", str(limit),
        ]

        if cover and Path(cover).exists():
            cmd.extend(["--cover", cover])
        if dynamic:
            cmd.extend(["--dynamic", dynamic])
        if dtime:
            cmd.extend(["--dtime", str(int(dtime))])
        if collection_id is not None:
            cmd.extend(["--extra-fields", json.dumps({"season_id": collection_id})])

        logger.info(f"📤 Uploading to Bilibili: {upload_title}")
        result = self._run(cmd, timeout=600)

        if result.returncode != 0:
            err = result.stderr or result.stdout or "Unknown error"
            raise RuntimeError(f"Bilibili upload failed: {err}")

        # Try to extract bvid from output
        bvid = self._extract_bvid(result.stdout + result.stderr)
        if bvid:
            logger.success(f"✅ Bilibili upload success: {bvid}")
            return bvid

        # If no bvid found, return a placeholder
        logger.info("✅ Bilibili upload completed (bvid not found in output)")
        return "uploaded"

    def get_collections(self) -> List[Dict[str, int]]:
        """
        Fetch user's Bilibili collections (seasons) list.

        Returns:
            List of dicts with 'season_id' and 'name' keys.
        """
        try:
            cookie_data = json.loads(Path(self.cookie_path).read_text(encoding="utf-8"))
            cookies = cookie_data.get("cookie_info", {}).get("cookies", [])
            sessdata = next((c["value"] for c in cookies if c.get("name") == "SESSDATA"), None)
            mid = next((c["value"] for c in cookies if c.get("name") == "DedeUserID"), None)

            if not sessdata or not mid:
                logger.warning("SESSDATA or DedeUserID not found in cookie")
                return []

            url = f"https://api.bilibili.com/x/polymer/web-space/seasons_series_list?mid={mid}&page_num=1&page_size=50"
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                "Referer": f"https://space.bilibili.com/{mid}",
                "Cookie": f"SESSDATA={sessdata}",
            })
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())

            if data.get("code") != 0:
                logger.warning(f"Failed to fetch collections: {data.get('message')}")
                return []

            seasons = data.get("data", {}).get("items_lists", {}).get("seasons_list", [])
            collections = []
            for s in seasons:
                meta = s.get("meta", {})
                season_id = meta.get("season_id")
                name = meta.get("name")
                if season_id and name:
                    collections.append({"season_id": season_id, "name": name})
            return collections
        except Exception as e:
            logger.warning(f"Failed to fetch collections: {e}")
            return []

    @staticmethod
    def _extract_bvid(text: str) -> Optional[str]:
        """Extract BVid from biliup output."""
        # Common patterns: BV1xx411c7mD
        match = re.search(r"(BV[0-9A-Za-z]{10})", text)
        if match:
            return match.group(1)
        # Also try bvid=xxx or https://www.bilibili.com/video/BVxxx
        match = re.search(r"bilibili\.com/video/(BV[0-9A-Za-z]{10})", text)
        if match:
            return match.group(1)
        return None
