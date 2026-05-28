from mininet.net import Mininet
from mininet.topo import SingleSwitchTopo  # 可以根据需要更改拓扑
from time import sleep


def run_pingall(net, interval):
    """
    定时执行 pingall 命令
    :param net: Mininet 网络实例
    :param interval: 执行 pingall 操作的时间间隔（秒）
    """
    while True:
        print("Running pingall...")
        net.pingAll()  # 执行 pingall 操作
        sleep(interval)  # 暂停一段时间


if __name__ == '__main__':
    # 创建一个简单的拓扑，包含一个交换机和 2 个主机
    topo = SingleSwitchTopo(2)

    # 创建并启动网络
    net = Mininet(topo=topo)
    net.start()

    # 定时每 5 秒运行一次 pingall
    run_pingall(net, 5)

    # 停止网络
    net.stop()