# 通过链路信息（字典）来统计某一device的端口数(不含主机端口)
def count_port(links_data, device):
    num = 0
    visited = []
    for i in range(len(links_data)):
        if links_data[i]["src"]["device"] == device:
            if not links_data[i]["src"]["port"] in visited:
                num += 1
                visited.append(links_data[i]["src"]["port"])
        if links_data[i]["dst"]["device"] == device:
            if not links_data[i]["dst"]["port"] in visited:
                num += 1
                visited.append(links_data[i]["dst"]["port"])

    return num

# 利用链路信息（字典）生成端口数列表（含主机端口）
def build_port_array(links_data, devices_data, hosts_data):
    port_array = [0]*len(devices_data)
    idx = 0
    for device in devices_data:
        port_num = count_port(links_data, device)
        port_array[idx] = port_num
        idx += 1
    port_array[hosts_data["00:00:00:00:00:00:01/None"][0] - 1] += 1
    port_array[hosts_data["00:00:00:00:00:00:02/None"][0] - 1] += 1
    return port_array
#test1
dic = {
  "links": [
    {
      "src": {
        "port": "2",
        "device": "of:0000000000000001"
      },
      "dst": {
        "port": "1",
        "device": "of:0000000000000004"
      },
      "type": "DIRECT",
      "state": "ACTIVE"
    },


    {
      "src": {
        "port": "3",
        "device": "of:0000000000000001"
      },
      "dst": {
        "port": "3",
        "device": "of:0000000000000003"
      },
      "type": "DIRECT",
      "state": "ACTIVE"
    },
    {
      "src": {
        "port": "3",
        "device": "of:0000000000000002"
      },
      "dst": {
        "port": "3",
        "device": "of:0000000000000005"
      },
      "type": "DIRECT",
      "state": "ACTIVE"
    },
    {
      "src": {
        "port": "2",
        "device": "of:0000000000000003"
      },
      "dst": {
        "port": "2",
        "device": "of:0000000000000005"
      },
      "type": "DIRECT",
      "state": "ACTIVE"
    },
    {
      "src": {
        "port": "4",
        "device": "of:0000000000000001"
      },
      "dst": {
        "port": "2",
        "device": "of:0000000000000002"
      },
      "type": "DIRECT",
      "state": "ACTIVE"
    },
    {
      "src": {
        "port": "1",
        "device": "of:0000000000000004"
      },
      "dst": {
        "port": "2",
        "device": "of:0000000000000001"
      },
      "type": "DIRECT",
      "state": "ACTIVE"
    },
    {
      "src": {
        "port": "3",
        "device": "of:0000000000000003"
      },
      "dst": {
        "port": "3",
        "device": "of:0000000000000001"
      },
      "type": "DIRECT",
      "state": "ACTIVE"
    },
    {
      "src": {
        "port": "3",
        "device": "of:0000000000000005"
      },
      "dst": {
        "port": "3",
        "device": "of:0000000000000002"
      },
      "type": "DIRECT",
      "state": "ACTIVE"
    },
    {
      "src": {
        "port": "2",
        "device": "of:0000000000000002"
      },
      "dst": {
        "port": "4",
        "device": "of:0000000000000001"
      },
      "type": "DIRECT",
      "state": "ACTIVE"
    },
    {
      "src": {
        "port": "2",
        "device": "of:0000000000000005"
      },
      "dst": {
        "port": "2",
        "device": "of:0000000000000003"
      },
      "type": "DIRECT",
      "state": "ACTIVE"
    },
    {
      "src": {
        "port": "1",
        "device": "of:0000000000000005"
      },
      "dst": {
        "port": "2",
        "device": "of:0000000000000004"
      },
      "type": "DIRECT",
      "state": "ACTIVE"
    },
    {
      "src": {
        "port": "2",
        "device": "of:0000000000000004"
      },
      "dst": {
        "port": "1",
        "device": "of:0000000000000005"
      },
      "type": "DIRECT",
      "state": "ACTIVE"
    }
  ]
}
devices_data = ["of:0000000000000001", "of:0000000000000002", "of:0000000000000003",
               "of:0000000000000004", "of:0000000000000005"]
hosts_data = {"00:00:00:00:00:00:01/None": [1, 1], "00:00:00:00:00:00:02/None": [5, 4]}
port_array = build_port_array(dic["links"], devices_data, hosts_data)
print(port_array)

# 通过lable来推算端口信息（含主机端口）
def find_port(port_array, n):
    # hosts_data = {"h1": [1, 4], "h2": [5, 4]}
    n += 1
    for i in range(len(port_array)):
        n -= port_array[i]
        if n <= 0:
            n += port_array[i]
            return [i+1, n]

# 通过端口信息来推算lable（含主机端口）
def find_lable(port_array, port):
    lable = -1
    for i in range(port[0]-1):
        lable += port_array[i]
    return lable + port[1]

#test2
port = find_port(port_array, 0)
lable = find_lable(port_array, [1, 1])
print(port, lable)

# 通过交换机信息把mac地址换算为交换机号
def tran_mac(devices_data, device):
    num = 1
    for i in range(len(devices_data)):
        if device == devices_data[i]:
          return num
        num += 1

# 读取src的lable
def read_src(links_data, devices_data, port_array, i):
    src_port = links_data[i]["src"]["port"]
    src_mac = links_data[i]["src"]["device"]
    src_device_num = tran_mac(devices_data, src_mac)
    src = [src_device_num, int(src_port)]
    return find_lable(port_array, src)

# 读取dst的lable
def read_dst(links_data, devices_data, port_array, i):
    dst_port = links_data[i]["dst"]["port"]
    dst_mac = links_data[i]["dst"]["device"]
    dst_device_num = tran_mac(devices_data, dst_mac)
    dst = [dst_device_num, int(dst_port)]
    return find_lable(port_array, dst)

# 通过链路信息(字典)构造二维数组
def build_link_array(links_data, devices_data, port_array):
    total = sum(port_array)
    link_array = [[100 for _ in range(total)] for _ in range(total)]
    for i in range(len(links_data)):
        if links_data[i]["type"] == "DIRECT" and links_data[i]["state"] == "ACTIVE":
          src = read_src(links_data, devices_data, port_array, i)
          dst = read_dst(links_data, devices_data, port_array, i)
          link_array[src][dst] = 1
    acc = 0
    for i in range(len(devices_data)):
        num = port_array[i]
        for j in range(num):
            link_array[acc+j][acc:acc+num] = [0 for _ in range(num)]
        acc += port_array[i]
    return link_array

# test3
link_array = build_link_array(dic["links"], devices_data, port_array)
for lst in link_array:
    print(lst)

