from manim import *


class BlockMaskUpdate(Scene):
    def construct(self):
        # 【背景设置】纯白背景
        self.camera.background_color = WHITE

        L = 8
        # 【颜色配置】
        COLOR_ACTIVE = "#F51E1E"  # 深红色 (更新节点)
        COLOR_FROZEN = "#489696"  # 白红色 (冻结节点)

        # 【排版参数】
        spacing = 0.55  # 格点间距
        dot_radius = spacing / 10.0  # 圆点半径

        # ==========================================
        # 1. 绘制底层黑色网格线与格点
        # ==========================================
        grid_lines = VGroup()
        dots_group = VGroup()
        dots = {}

        for i in range(L):
            coord = (i - (L - 1) / 2) * spacing
            min_coord = (0 - (L - 1) / 2) * spacing
            max_coord = (L - 1 - (L - 1) / 2) * spacing

            h_line = Line(start=RIGHT * min_coord + DOWN * coord, end=RIGHT * max_coord + DOWN * coord, color=BLACK,
                          stroke_width=1.5)
            v_line = Line(start=RIGHT * coord + DOWN * min_coord, end=RIGHT * coord + DOWN * max_coord, color=BLACK,
                          stroke_width=1.5)
            grid_lines.add(h_line, v_line)

        for y in range(L):
            for x in range(L):
                dot = Dot(color=COLOR_FROZEN, radius=dot_radius)
                pos_x = (x - (L - 1) / 2) * spacing
                pos_y = (y - (L - 1) / 2) * spacing
                dot.move_to(RIGHT * pos_x + DOWN * pos_y)
                dots_group.add(dot)
                dots[(x, y)] = dot

        # 【位置控制】将整个晶格向下平移 0.8 单位，确保顶部留白
        lattice = VGroup(grid_lines, dots_group).shift(DOWN * 0.8)

        self.play(FadeIn(lattice, run_time=1.0))

        # ==========================================
        # 2. 文字坐标系统 (绝对坐标)
        # ==========================================
        TITLE_POS = UP * 3.2  # 主标题固定高度
        STATUS_POS = UP * 2.5  # 副标题固定高度

        # ==========================================
        # 3. 核心循环：遍历 block_num = [2, 4, 8]
        # ==========================================
        block_nums = [2, 4, 8]

        for b_num in block_nums:
            block_size = L // b_num

            # --- 处理主标题 ---
            title = Text(f"Mask Pattern: block_num = {b_num} ({block_size}x{block_size} Blocks)",
                         font_size=32, color=BLACK).move_to(TITLE_POS)
            self.play(Write(title, run_time=0.6))

            # --- 内部 Step 循环 ---
            for step in range(1, 4):
                is_even_active = (step % 2 == 1)

                # --- 处理副标题 (Step 描述) ---
                # 每次创建新对象并 FadeIn
                status = Text(f"Step {step}: Updating {'Even' if is_even_active else 'Odd'} Blocks",
                              font_size=24, color=BLACK).move_to(STATUS_POS)
                self.play(FadeIn(status, run_time=0.4))

                # --- 计算圆点颜色变化 ---
                animations = []
                for y in range(L):
                    for x in range(L):
                        bx, by = x // block_size, y // block_size
                        is_even_block = ((bx + by) % 2 == 0)

                        target_color = COLOR_ACTIVE if (is_even_active == is_even_block) else COLOR_FROZEN
                        animations.append(dots[(x, y)].animate.set_color(target_color))

                # 【时长控制】1.5秒渐变 + 0.8秒展示
                self.play(*animations, run_time=1.5)
                self.wait(0.8)

                # --- 重要：在进入下一个 Step 前，清理掉当前的副标题 ---
                self.play(FadeOut(status, run_time=0.3))

            # --- 当前 block_num 演示结束，清理主标题 ---
            self.play(FadeOut(title, run_time=0.5))
            # 重置颜色为冻结色，准备进入下一个模式
            self.play(*[d.animate.set_color(COLOR_FROZEN) for d in dots.values()], run_time=0.5)

        self.play(FadeOut(lattice))
        self.wait(0.5)