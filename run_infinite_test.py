import time
import itertools  # 用于双流测试

# 导入我们定义的所有工具
from signal_processing import (
    SignalSequence, 
    infinite_source,
    gen_delay, gen_diff, gen_accum, gen_add, gen_convolve_finite,
    gen_mul, gen_upsample, gen_downsample, 
    gen_sliding_window_similarity_causal, gen_normalized_similarity
)

# 1. 创建从 "命令" -> "函数" 的映射
OPS_SINGLE_STREAM = {
    'diff': gen_diff,
    'accum': gen_accum,
    'delay': gen_delay,
    'upsample': gen_upsample,
    'downsample': gen_downsample,
}
OPS_MULTI_STREAM_TEMPLATE = {
    'convolve': gen_convolve_finite,
    'similarity': gen_sliding_window_similarity_causal,
    'ncc': gen_normalized_similarity,
}
OPS_DUAL_STREAM = {
    'add': gen_add,
    'mul': gen_mul,
}

# ====================================================================
# 模式 1: 单流管线构建器 (测试 3a, 3b, 3d, 3e, 4c, 4d)
# ====================================================================

def build_single_stream_pipeline():
    """
    一个交互式管线构建器 (用于 1 个输入流)
    """
    
    pipeline_stages = []
    
    print("--- 欢迎来到“单流管线”构建器 ---")
    print("您将从 [输入] 开始构建一个处理管线。")
    print("\n--- 可用的“单序列”操作 ---")
    print("  diff             (差分)")
    print("  accum            (累加)")
    print("  delay <k>        (延迟 k 个样本, 例如: delay 3)")
    print("  upsample <k>     (上采样 k 倍, 例如: upsample 2)")
    print("  downsample <k>   (下采样 k 倍, 例如: downsample 2)")
    print("\n--- 可用的“多序列”操作 (流 + 模板) ---")
    print("  convolve         (与 h[n] 卷积)")
    print("  ncc              (归一化互相关)")
    print("  ---")
    print("  run              (完成构建并开始运行)")
    print("  quit             (退出)\n")
    
    while True:
        # 1. 获取用户输入
        current_pipe_str = ' -> '.join([s[0] for s in pipeline_stages])
        raw_input = input(f"当前管线: [输入] -> {current_pipe_str} -> [输出]\n请添加操作 (或 'run'): ")
        parts = raw_input.strip().lower().split()
        
        if not parts: continue
        cmd = parts[0]
        
        # --- 控制命令 ---
        if cmd == 'run': break
        if cmd == 'quit': return

        # --- 单序列操作 ---
        if cmd in OPS_SINGLE_STREAM:
            func = OPS_SINGLE_STREAM[cmd]
            
            # 检查是否需要参数 (k)
            if cmd in ('delay', 'upsample', 'downsample'):
                if len(parts) < 2:
                    print(f"错误: {cmd} 需要一个数字参数 (例如: {cmd} 2)")
                    continue
                try:
                    k = int(parts[1])
                    pipeline_stages.append((f"{cmd}({k})", func, [k])) # (名称, 函数, [参数])
                    print(f"  -> 已添加: {cmd}(k={k})")
                except ValueError:
                    print(f"错误: '{parts[1]}' 不是一个有效的数字。")
            else:
                # 不需要参数 (diff, accum)
                pipeline_stages.append((cmd, func, []))
                print(f"  -> 已添加: {cmd}")
        
        # --- 多序列操作 (带模板) ---
        elif cmd in OPS_MULTI_STREAM_TEMPLATE:
            print(f"\n--- 配置 {cmd} ---")
            print("这需要一个有限的 h[n] 模板。")
            h_data_str = input("请输入 h 的数据 (用空格分隔, 例如: 1 2 1): ")
            h_n0_str = input("请输入 h 的起始索引 n0 (例如: 0): ")
            try:
                h_data = [float(s) for s in h_data_str.split()]
                h_n0 = int(h_n0_str)
                h_seq = SignalSequence(h_data, h_n0)
                
                func = OPS_MULTI_STREAM_TEMPLATE[cmd]
                pipeline_stages.append((f"{cmd}(h={h_data})", func, [h_seq]))
                print(f"  -> 已添加: {cmd} (模板={h_seq})")
            except Exception as e:
                print(f"错误: 无法创建模板: {e}")
        
        else:
            print(f"错误: 未知的命令 '{cmd}'")

    # --- 2. 构建管线 ---
    if not pipeline_stages:
        print("管线为空。将直接输出源。")
    
    print("\n--- 正在构建管线... ---")
    pipeline = infinite_source()
    
    for (name, func, args) in pipeline_stages:
        print(f"  ...包裹 {name}")
        pipeline = func(pipeline, *args) 

    # --- 3. 运行管线 ---
    print("\n--- 管线已建立，等待输入... ---")
    try:
        for final_output in pipeline:
            print(f"  ==> [管线最终输出]: {final_output:.4f}")
    except KeyboardInterrupt:
        print("\n--- 用户强制退出 ---")


# ====================================================================
# 模式 2: 双流操作测试 (测试 4a, 4b)
# ====================================================================

def test_dual_stream_pipeline():
    """
    一个用于测试 `gen_add` 和 `gen_mul` 的专用演示。
    它将 (用户输入流) 与 (一个固定的周期流) 相结合。
    """
    print("--- 欢迎来到“双流操作”测试仪 ---")
    print("您将提供 [流 1 (输入)]。")
    print("[流 2] 将是一个固定的周期序列: [10, 20, 30]")
    
    cmd = input("请选择操作 ('add' 或 'mul'): ").strip().lower()
    
    if cmd not in OPS_DUAL_STREAM:
        print("无效操作。")
        return
        
    operation_func = OPS_DUAL_STREAM[cmd]
    
    # 1. 创建流 1 (用户输入)
    stream1 = infinite_source()
    
    # 2. 创建流 2 (固定流)
    stream2 = itertools.cycle([10.0, 20.0, 30.0])
    
    # 3. 构建管线
    # pipeline = gen_add(stream1, stream2)
    pipeline = operation_func(stream1, stream2)
    
    print(f"\n--- 管线已建立: [您的输入] {cmd} [10, 20, 30, ...] ---")
    try:
        for final_output in pipeline:
            print(f"  ==> [管线最终输出]: {final_output:.4f}")
    except KeyboardInterrupt:
        print("\n--- 用户强制退出 ---")


# ====================================================================
# 主菜单
# ====================================================================

def main_menu():
    print("--- “即时操作”测试程序 ---")
    print("请选择要测试的类别：")
    print("\n1: [单流管线] (构建器)")
    print("   (测试: diff, accum, delay, upsample, convolve, ncc 等)")
    print("\n2: [双流操作] (演示)")
    print("   (测试: add, mul)")
    
    choice = input("\n请输入您的选择 (1 或 2): ").strip()
    
    if choice == '1':
        build_single_stream_pipeline()
    elif choice == '2':
        test_dual_stream_pipeline()
    else:
        print("无效选择。退出。")

# --- 主程序入口 ---
if __name__ == "__main__":
    main_menu()