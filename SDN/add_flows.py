import requests
from requests.auth import HTTPBasicAuth
from requests_toolbelt.utils.formdata import urlencode
from zmq.backend.cffi import device

onos_host = "localhost"
auth = HTTPBasicAuth("karaf","karaf")
app_id = 0  #定义的全局变量，代表下发的流表

#用于清理之前下发的流表
def clear_flows_by_app_id(controller_ip):
    global app_id
    resp = []
    while app_id != 0:
        headers = {'Content-Type': 'application/json'}
        url = f"http://{controller_ip}:8181/onos/v1/flows/application/{app_id}"
        resp.append(requests.delete(url=url, headers=headers, auth=auth))
        app_id -= 1
    return resp

#test数据
result_dijkstra = {
    "of:0000000000000001": {
        "src_port": "2",
        "dst_port": "1",
    },
    "of:0000000000000003": {
        "src_port": "1",
        "dst_port": "2",
    },
    "of:0000000000000005": {
        "src_port": "3",
        "dst_port": "2",
    }
}
#根据路径安排的算法下发流表
def add_flows_by_dijkstra(controller_ip, result_dijkstra):
    headers = {"Content-Type": "application/json", "Accept": "appliction/json"}
    global app_id
    resp = []
    for device_id in result_dijkstra:   #循环，为路径上每一个交换机下发流表
        url = f"http://{controller_ip}:8181/onos/v1/flows/{device_id}"
        app_id += 1
        params = {"appId": app_id}
        data = {
            "priority": 40000,
            "timeout": 0,
            "isPermanent": True,    #这里似乎可以设置为False，但不知道到会持续多长时间？？
            "deviceId":device_id,
            "treatment": {"instructions": [{"type": "OUTPUT", "port": device_id["dst_port"]}]},
            "selector": {"criteria": [{"type": "INPUT", "port": device_id["src_port"]}]},
        }
        resp.append(requests.post(url=url, headers=headers, params=params, json=data, auth=auth))
    return resp

