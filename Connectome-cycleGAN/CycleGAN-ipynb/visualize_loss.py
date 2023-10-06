import matplotlib.pyplot as plt
import pandas as pd
import re

def parse_line(line):
    pattern = r'\(epoch: (\d+), iters: \d+, time: [\d\.]+, data: [\d\.]+\) D_A: ([\d\.]+) G_A: ([\d\.]+) cycle_A: ([\d\.]+) idt_A: [\d\.]+ D_B: ([\d\.]+) G_B: ([\d\.]+) cycle_B: ([\d\.]+) idt_B: [\d\.]+'
    match = re.search(pattern, line)
    if match:
        epoch, d_a, g_a, cycle_a, d_b, g_b, cycle_b = map(float, match.groups())
        return epoch, d_a, g_a, cycle_a, d_b, g_b, cycle_b
    else:
        return None

def read_and_plot(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    data = [parse_line(line) for line in lines if parse_line(line) is not None]
    df = pd.DataFrame(data, columns=['Epoch', 'D_A', 'G_A', 'Cycle_A', 'D_B', 'G_B', 'Cycle_B'])
    df.set_index('Epoch', inplace=True)

    plt.figure(figsize=(20, 10))

    plt.subplot(231)
    plt.plot(df['D_A'], label='D_A')
    plt.xlabel('Epoch')
    plt.ylabel('D_A')
    plt.legend()

    plt.subplot(232)
    plt.plot(df['G_A'], label='G_A')
    plt.xlabel('Epoch')
    plt.ylabel('G_A')
    plt.legend()

    plt.subplot(233)
    plt.plot(df['Cycle_A'], label='Cycle_A')
    plt.xlabel('Epoch')
    plt.ylabel('Cycle_A')
    plt.legend()

    plt.subplot(234)
    plt.plot(df['D_B'], label='D_B')
    plt.xlabel('Epoch')
    plt.ylabel('D_B')
    plt.legend()

    plt.subplot(235)
    plt.plot(df['G_B'], label='G_B')
    plt.xlabel('Epoch')
    plt.ylabel('G_B')
    plt.legend()

    plt.subplot(236)
    plt.plot(df['Cycle_B'], label='Cycle_B')
    plt.xlabel('Epoch')
    plt.ylabel('Cycle_B')
    plt.legend()

    plt.tight_layout()
    plt.show()

read_and_plot("/nfs2/xuh11/Connectome/CycleGAN-ipynb/pytorch-CycleGAN-and-pix2pix/checkpoints/BIOCARDtoVMAP_Mean_Length_1/loss_log_2.txt")