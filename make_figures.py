from plot_utils import ExperimentVisualizer


# 初始化
vis = ExperimentVisualizer(default_figsize=(8, 6)) # 论文标准比例

# 定义一个统一的“稀疏虚线”样式，方便随时微调
sparse_dashed = (0, (5, 8)) # 5长实线，8长空白

# 载入数据：只需指定 group，颜色就会自动分配（同一组同色）
# 也可以像最后一行那样手动指定颜色
vis.add_data('Block 2', 'cnn_res_block_mask_model_double_precision_False_L_8_blockNum_2_coupling_10_hidden_6_channels_16_iterations_15000_loss_history.npy', group='B2')
vis.add_data('pre-processing Block 2', 'prior_cnn_res_block_mask_double_precision_False_L_8_blockNum_2_coupling_10_hidden_6_channels_16_iterations_15000_loss_history.npy', group='B2', linestyle=sparse_dashed)

vis.add_data('Block 4', 'cnn_res_block_mask_model_double_precision_False_L_8_blockNum_4_coupling_10_hidden_6_channels_16_iterations_15000_loss_history.npy', group='B4')
vis.add_data('pre-processing Block 4', 'prior_cnn_res_block_mask_double_precision_False_L_8_blockNum_4_coupling_10_hidden_6_channels_16_iterations_15000_loss_history.npy', group='B4', linestyle=sparse_dashed)

vis.add_data('Block 8', 'cnn_res_block_mask_model_double_precision_False_L_8_blockNum_8_coupling_10_hidden_6_channels_16_iterations_15000_loss_history.npy', group='B8', color='green') # 手动指定色系
vis.add_data('pre-processing Block 8', 'prior_cnn_res_block_mask_double_precision_False_L_8_blockNum_8_coupling_10_hidden_6_channels_16_iterations_15000_loss_history.npy', group='B8', linestyle=sparse_dashed)

# 绘图逻辑
vis.create_subplots(1, 1)
vis.plot_on_ax(ax_index=0,
               title='Different Block Loss Comparison(pre-processing vs none-pre-processing)',
               smooth=True,        # 开启平滑
               smooth_weight=0.96,   # 平滑强度
               show_raw=True,      # 叠加原始阴影
               y_lim=(-16, 0),
               x_start=5)     # 坐标轴缩放

vis.show()

# 如果窗口关了你还想用脚本保存一份 PDF：
# vis.save('my_paper_plot.pdf')