import os
import pygame


def _match_font(names):
    for name in names:
        try:
            path = pygame.font.match_font(name)
        except Exception:
            path = None
        if path:
            return path
    return None


def _find_local_font_dir():
    base = os.path.join(os.path.dirname(__file__), 'fonts')
    return os.path.abspath(base)


def _pick_local_font():
    dir_path = _find_local_font_dir()
    if not os.path.exists(dir_path):
        return None
    for f in os.listdir(dir_path):
        if f.lower().endswith(('.ttf', '.otf')):
            return os.path.join(dir_path, f)
    return None


def load_cn_font(size=16):
    """加载中文兼容字体，优先系统中文字体，其次本地fonts目录，最后默认字体。"""
    pygame.font.init()
    # 常见中文字体名称（macOS / Windows / Noto）
    candidates = [
        'PingFang SC', 'PingFangSC-Regular', 'Hiragino Sans GB', 'Heiti SC', 'STHeiti', 'Songti SC',
        'Noto Sans CJK SC', 'NotoSansCJKsc-Regular', 'Microsoft YaHei', 'SimHei', 'SimSun',
        'Arial Unicode MS'
    ]
    path = _match_font(candidates)
    if not path:
        path = _pick_local_font()
    if path:
        try:
            return pygame.font.Font(path, size)
        except Exception:
            pass
    # 兜底：系统默认字体（可能中文不完整）
    return pygame.font.SysFont(None, size)