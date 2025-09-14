import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import tushare as ts


def _load_dotenv_if_exists() -> None:
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        # Fallback to default .env discovery from CWD and parents
        load_dotenv()


def get_tushare_pro(token: Optional[str] = None):
    """
    返回已初始化的 Tushare Pro API 实例。
    优先从入参 token 读取，否则从 .env / 环境变量读取 TUSHARE_TOKEN。
    """
    _load_dotenv_if_exists()

    resolved_token = token or os.getenv("TUSHARE_TOKEN") or os.getenv("TS_TOKEN") or os.getenv("TUSHARE_PRO_TOKEN")
    if not resolved_token:
        raise RuntimeError(
            "未找到 TUSHARE_TOKEN。请在项目根目录创建 .env，并设置 TUSHARE_TOKEN=你的token"
        )

    ts.set_token(resolved_token)
    return ts.pro_api(resolved_token)
