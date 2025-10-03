import os
import sys
import pygame


def main():
    # 将仓库根目录加入 sys.path 以便导入 src 模块
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from src.ui.font_loader import load_cn_font

    pygame.init()
    surface = pygame.Surface((900, 240))
    surface.fill((30, 40, 50))

    font_big = load_cn_font(48)
    font_small = load_cn_font(22)

    lines = [
        "中文字体预览：海战 游戏 地图 显示",
        "地图: ground_based_platforms.png",
        "FPS: 60  状态: 运行中"
    ]

    y = 20
    for i, text in enumerate(lines):
        font = font_big if i == 0 else font_small
        img = font.render(text, True, (255, 255, 255))
        surface.blit(img, (20, y))
        y += img.get_height() + 14

    out_path = os.path.join(os.path.dirname(__file__), 'probe.png')
    pygame.image.save(surface, out_path)
    print(f"Saved preview to: {out_path}")


if __name__ == '__main__':
    main()