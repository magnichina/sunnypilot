#!/usr/bin/env python3
"""
纵向模型预测控制器 (Longitudinal MPC)
用于自动驾驶车辆的纵向控制，包括自适应巡航控制(ACC)和混合模式控制
"""

import os
import time
import numpy as np
from cereal import log
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.realtime import DT_MDL
from openpilot.common.swaglog import cloudlog
# WARNING: imports outside of constants will not trigger a rebuild
from openpilot.selfdrive.modeld.constants import index_function
from openpilot.selfdrive.controls.radard import _LEAD_ACCEL_TAU

# 导入ACADOS优化求解器相关模块
if __name__ == '__main__':  # generating code
  from openpilot.third_party.acados.acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
else:
  from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.c_generated_code.acados_ocp_solver_pyx import AcadosOcpSolverCython

from casadi import SX, vertcat

# 模型和文件路径配置
MODEL_NAME = 'long'  # 模型名称
LONG_MPC_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(LONG_MPC_DIR, "c_generated_code")  # C代码生成目录
JSON_FILE = os.path.join(LONG_MPC_DIR, "acados_ocp_long.json")

# 控制源类型定义
SOURCES = ['lead0', 'lead1', 'cruise', 'e2e']  # 前车0、前车1、巡航、端到端

# 维度定义
X_DIM = 3      # 状态维度：位置、速度、加速度
U_DIM = 1      # 控制维度：加加速度（jerk）
PARAM_DIM = 6  # 参数维度
COST_E_DIM = 5 # 终端代价维度
COST_DIM = COST_E_DIM + 1
CONSTR_DIM = 4 # 约束维度

# 代价函数权重配置
X_EGO_OBSTACLE_COST = 3.    # 与前车距离的代价权重
X_EGO_COST = 0.            # 自身位置的代价权重
V_EGO_COST = 0.            # 自身速度的代价权重
A_EGO_COST = 0.            # 自身加速度的代价权重
J_EGO_COST = 5.0           # 加加速度的代价权重
A_CHANGE_COST = 200.       # 加速度变化的代价权重
DANGER_ZONE_COST = 100.    # 危险区域代价权重
CRASH_DISTANCE = .25       # 碰撞距离阈值
LEAD_DANGER_FACTOR = 0.75  # 前车危险因子
LIMIT_COST = 1e6           # 约束违反的代价权重
ACADOS_SOLVER_TYPE = 'SQP_RTI'  # 求解器类型

# 预测时域配置
# 较少的时间戳不会影响性能，并且在低迭代次数下能获得更好的MPC收敛性
N = 12        # 预测步数
MAX_T = 10.0  # 最大预测时间（秒）
T_IDXS_LST = [index_function(idx, max_val=MAX_T, max_idx=N) for idx in range(N+1)]

T_IDXS = np.array(T_IDXS_LST)  # 时间索引数组
FCW_IDXS = T_IDXS < 5.0        # 前向碰撞预警时间索引
T_DIFFS = np.diff(T_IDXS, prepend=[0.])  # 时间间隔

# 纵向控制参数
COMFORT_BRAKE = 2.2      # 舒适制动减速度 (m/s²)
STOP_DISTANCE = 4.0      # 停止距离 (米)
CRUISE_MIN_ACCEL = -1.2  # 巡航最小加速度 (m/s²)
CRUISE_MAX_ACCEL = 1.6   # 巡航最大加速度 (m/s²)

def get_jerk_factor(personality=log.LongitudinalPersonality.standard):
    """根据驾驶个性获取加加速度因子"""
    if personality == log.LongitudinalPersonality.relaxed:
        return 1.0
    elif personality == log.LongitudinalPersonality.standard:
        return 1.0
    elif personality == log.LongitudinalPersonality.aggressive:
        return 0.5  # 激进模式下允许更大的加加速度
    else:
        raise NotImplementedError("Longitudinal personality not supported")

def get_T_FOLLOW(personality=log.LongitudinalPersonality.standard):
    """根据驾驶个性获取跟车时间间隔"""
    if personality == log.LongitudinalPersonality.relaxed:
        return 1.75  # 放松模式：较长的跟车距离
    elif personality == log.LongitudinalPersonality.standard:
        return 1.45  # 标准模式
    elif personality == log.LongitudinalPersonality.aggressive:
        return 1.25  # 激进模式：较短的跟车距离
    else:
        raise NotImplementedError("Longitudinal personality not supported")

def get_stopped_equivalence_factor(v_lead):
    """计算前车停止等效距离因子"""
    return (v_lead**2) / (2 * COMFORT_BRAKE)

def get_safe_obstacle_distance(v_ego, t_follow):
    """计算安全障碍物距离"""
    return (v_ego**2) / (2 * COMFORT_BRAKE) + t_follow * v_ego + STOP_DISTANCE

def desired_follow_distance(v_ego, v_lead, t_follow=None):
    """计算期望跟车距离"""
    if t_follow is None:
        t_follow = get_T_FOLLOW()
    return get_safe_obstacle_distance(v_ego, t_follow) - get_stopped_equivalence_factor(v_lead)

def gen_long_model():
    """生成纵向动力学模型"""
    model = AcadosModel()
    model.name = MODEL_NAME

    # 设置状态变量：位置、速度、加速度
    x_ego = SX.sym('x_ego')
    v_ego = SX.sym('v_ego')
    a_ego = SX.sym('a_ego')
    model.x = vertcat(x_ego, v_ego, a_ego)

    # 控制变量：加加速度
    j_ego = SX.sym('j_ego')
    model.u = vertcat(j_ego)

    # 状态导数
    x_ego_dot = SX.sym('x_ego_dot')
    v_ego_dot = SX.sym('v_ego_dot')
    a_ego_dot = SX.sym('a_ego_dot')
    model.xdot = vertcat(x_ego_dot, v_ego_dot, a_ego_dot)

    # 实时参数
    a_min = SX.sym('a_min')  # 最小加速度
    a_max = SX.sym('a_max')  # 最大加速度
    x_obstacle = SX.sym('x_obstacle')  # 障碍物位置
    prev_a = SX.sym('prev_a')  # 上一时刻加速度
    lead_t_follow = SX.sym('lead_t_follow')  # 跟车时间
    lead_danger_factor = SX.sym('lead_danger_factor')  # 危险因子
    model.p = vertcat(a_min, a_max, x_obstacle, prev_a, lead_t_follow, lead_danger_factor)

    # 动力学模型：x' = v, v' = a, a' = j
    f_expl = vertcat(v_ego, a_ego, j_ego)
    model.f_impl_expr = model.xdot - f_expl
    model.f_expl_expr = f_expl
    return model

def gen_long_ocp():
    """生成纵向最优控制问题"""
    ocp = AcadosOcp()
    ocp.model = gen_long_model()

    Tf = T_IDXS[-1]  # 终端时间

    # 设置维度
    ocp.dims.N = N  # 预测步数

    # 设置代价函数类型
    ocp.cost.cost_type = 'NONLINEAR_LS'  # 非线性最小二乘
    ocp.cost.cost_type_e = 'NONLINEAR_LS'  # 终端代价

    QR = np.zeros((COST_DIM, COST_DIM))
    Q = np.zeros((COST_E_DIM, COST_E_DIM))

    ocp.cost.W = QR  # 阶段代价权重矩阵
    ocp.cost.W_e = Q  # 终端代价权重矩阵

    # 提取状态和控制变量
    x_ego, v_ego, a_ego = ocp.model.x[0], ocp.model.x[1], ocp.model.x[2]
    j_ego = ocp.model.u[0]

    # 提取参数
    a_min, a_max = ocp.model.p[0], ocp.model.p[1]
    x_obstacle = ocp.model.p[2]
    prev_a = ocp.model.p[3]
    lead_t_follow = ocp.model.p[4]
    lead_danger_factor = ocp.model.p[5]

    ocp.cost.yref = np.zeros((COST_DIM, ))  # 阶段代价参考值
    ocp.cost.yref_e = np.zeros((COST_E_DIM, ))  # 终端代价参考值

    desired_dist_comfort = get_safe_obstacle_distance(v_ego, lead_t_follow)

    # 主要代价函数：与期望距离的偏差
    costs = [
        ((x_obstacle - x_ego) - (desired_dist_comfort)) / (v_ego + 10.),  # 距离偏差
        x_ego,  # 位置
        v_ego,  # 速度
        a_ego,  # 加速度
        a_ego - prev_a,  # 加速度变化
        j_ego   # 加加速度
    ]
    ocp.model.cost_y_expr = vertcat(*costs)
    ocp.model.cost_y_expr_e = vertcat(*costs[:-1])  # 终端代价不包含加加速度

    # 约束条件：速度、加速度和距离约束
    constraints = vertcat(
        v_ego,  # 速度约束
        (a_ego - a_min),  # 加速度下限
        (a_max - a_ego),  # 加速度上限
        ((x_obstacle - x_ego) - lead_danger_factor * (desired_dist_comfort)) / (v_ego + 10.)  # 安全距离约束
    )
    ocp.model.con_h_expr = constraints

    x0 = np.zeros(X_DIM)
    ocp.constraints.x0 = x0  # 初始状态
    ocp.parameter_values = np.array([-1.2, 1.2, 0.0, 0.0, get_T_FOLLOW(), LEAD_DANGER_FACTOR])

    # 运行时设置约束代价权重
    cost_weights = np.zeros(CONSTR_DIM)
    ocp.cost.zl = cost_weights
    ocp.cost.Zl = cost_weights
    ocp.cost.Zu = cost_weights
    ocp.cost.zu = cost_weights

    ocp.constraints.lh = np.zeros(CONSTR_DIM)  # 约束下限
    ocp.constraints.uh = 1e4 * np.ones(CONSTR_DIM)  # 约束上限
    ocp.constraints.idxsh = np.arange(CONSTR_DIM)  # 约束索引

    # 求解器配置
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'  # 使用HPIPM求解器
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'  # 高斯牛顿法近似Hessian矩阵
    ocp.solver_options.integrator_type = 'ERK'  # 显式龙格-库塔积分器
    ocp.solver_options.nlp_solver_type = ACADOS_SOLVER_TYPE
    ocp.solver_options.qp_solver_cond_N = 1

    # 迭代次数配置：更多迭代耗时，较少迭代在某些情况下收敛不准确
    ocp.solver_options.qp_solver_iter_max = 10  # 最大QP迭代次数
    ocp.solver_options.qp_tol = 1e-3  # QP容忍度

    # 设置预测时域
    ocp.solver_options.tf = Tf
    ocp.solver_options.shooting_nodes = T_IDXS

    ocp.code_export_directory = EXPORT_DIR
    return ocp

class LongitudinalMpc:
    """纵向模型预测控制器主类"""

    def __init__(self, mode='acc', dt=DT_MDL):
        """初始化MPC控制器"""
        self.mode = mode  # 控制模式：'acc'或'blended'
        self.dt = dt      # 时间步长
        self.solver = AcadosOcpSolverCython(MODEL_NAME, ACADOS_SOLVER_TYPE, N)
        self.reset()
        self.source = SOURCES[2]  # 默认控制源为巡航

    def reset(self):
        """重置控制器状态"""
        self.solver.reset()
        self.v_solution = np.zeros(N+1)  # 速度解
        self.a_solution = np.zeros(N+1)  # 加速度解
        self.prev_a = np.array(self.a_solution)  # 上一时刻加速度
        self.j_solution = np.zeros(N)    # 加加速度解
        self.yref = np.zeros((N+1, COST_DIM))  # 参考轨迹
        for i in range(N):
            self.solver.cost_set(i, "yref", self.yref[i])
        self.solver.cost_set(N, "yref", self.yref[N][:COST_E_DIM])
        self.x_sol = np.zeros((N+1, X_DIM))  # 状态解
        self.u_sol = np.zeros((N,1))         # 控制解
        self.params = np.zeros((N+1, PARAM_DIM))  # 参数
        for i in range(N+1):
            self.solver.set(i, 'x', np.zeros(X_DIM))
        self.last_cloudlog_t = 0  # 最后日志时间
        self.status = False       # 状态标志
        self.crash_cnt = 0.0      # 碰撞计数
        self.solution_status = 0  # 求解状态
        # 计时器
        self.solve_time = 0.0
        self.time_qp_solution = 0.0
        self.time_linearization = 0.0
        self.time_integrator = 0.0
        self.x0 = np.zeros(X_DIM)  # 初始状态
        self.set_weights()

    def set_cost_weights(self, cost_weights, constraint_cost_weights):
        """设置代价函数权重"""
        W = np.asfortranarray(np.diag(cost_weights))
        for i in range(N):
            # 在预测时域后期减少加速度变化的代价权重
            W[4,4] = cost_weights[4] * np.interp(T_IDXS[i], [0.0, 1.0, 2.0], [1.0, 1.0, 0.0])
            self.solver.cost_set(i, 'W', W)
        # 终端代价权重
        self.solver.cost_set(N, 'W', np.copy(W[:COST_E_DIM, :COST_E_DIM]))

        # 设置约束的L2松弛代价
        Zl = np.array(constraint_cost_weights)
        for i in range(N):
            self.solver.cost_set(i, 'Zl', Zl)

    def set_weights(self, prev_accel_constraint=True, personality=log.LongitudinalPersonality.standard):
        """根据控制模式和驾驶个性设置权重"""
        jerk_factor = get_jerk_factor(personality)
        if self.mode == 'acc':
            # ACC模式代价权重
            a_change_cost = A_CHANGE_COST if prev_accel_constraint else 0
            cost_weights = [X_EGO_OBSTACLE_COST, X_EGO_COST, V_EGO_COST, A_EGO_COST,
                           jerk_factor * a_change_cost, jerk_factor * J_EGO_COST]
            constraint_cost_weights = [LIMIT_COST, LIMIT_COST, LIMIT_COST, DANGER_ZONE_COST]
        elif self.mode == 'blended':
            # 混合模式代价权重
            a_change_cost = 40.0 if prev_accel_constraint else 0
            cost_weights = [0., 0.1, 0.2, 5.0, a_change_cost, 1.0]
            constraint_cost_weights = [LIMIT_COST, LIMIT_COST, LIMIT_COST, DANGER_ZONE_COST]
        else:
            raise NotImplementedError(f'Planner mode {self.mode} not recognized in planner cost set')
        self.set_cost_weights(cost_weights, constraint_cost_weights)

    def set_cur_state(self, v, a):
        """设置当前状态"""
        v_prev = self.x0[1]
        self.x0[1] = v  # 当前速度
        self.x0[2] = a  # 当前加速度
        # 如果速度变化过大，重新初始化所有状态
        if abs(v_prev - v) > 2.:  # 可能只在v < v_prev时有用
            for i in range(N+1):
                self.solver.set(i, 'x', self.x0)

    @staticmethod
    def extrapolate_lead(x_lead, v_lead, a_lead, a_lead_tau):
        """预测前车轨迹"""
        a_lead_traj = a_lead * np.exp(-a_lead_tau * (T_IDXS**2)/2.)  # 指数衰减的加速度
        v_lead_traj = np.clip(v_lead + np.cumsum(T_DIFFS * a_lead_traj), 0.0, 1e8)  # 速度轨迹
        x_lead_traj = x_lead + np.cumsum(T_DIFFS * v_lead_traj)  # 位置轨迹
        lead_xv = np.column_stack((x_lead_traj, v_lead_traj))  # 组合位置和速度
        return lead_xv

    def process_lead(self, lead):
        """处理前车信息"""
        v_ego = self.x0[1]
        if lead is not None and lead.status:
            # 使用真实前车数据
            x_lead = lead.dRel      # 相对距离
            v_lead = lead.vLead     # 前车速度
            a_lead = lead.aLeadK    # 前车加速度
            a_lead_tau = lead.aLeadTau  # 前车加速度时间常数
        else:
            # 模拟一个快速前车，使MPC能在相同模式下继续运行
            x_lead = 50.0
            v_lead = v_ego + 10.0
            a_lead = 0.0
            a_lead_tau = _LEAD_ACCEL_TAU

        # 如果预期立即碰撞，MPC将不会收敛
        # 将前车距离限制在仍可制动的范围内
        min_x_lead = ((v_ego + v_lead)/2) * (v_ego - v_lead) / (-ACCEL_MIN * 2)
        x_lead = np.clip(x_lead, min_x_lead, 1e8)
        v_lead = np.clip(v_lead, 0.0, 1e8)
        a_lead = np.clip(a_lead, -10., 5.)
        lead_xv = self.extrapolate_lead(x_lead, v_lead, a_lead, a_lead_tau)
        return lead_xv

    def update(self, radarstate, v_cruise, x, v, a, j, personality=log.LongitudinalPersonality.standard):
        """更新MPC控制器"""
        t_follow = get_T_FOLLOW(personality)
        v_ego = self.x0[1]
        self.status = radarstate.leadOne.status or radarstate.leadTwo.status  # 前车状态

        # 处理两个前车
        lead_xv_0 = self.process_lead(radarstate.leadOne)
        lead_xv_1 = self.process_lead(radarstate.leadTwo)

        # 估计移动前车的安全距离：计算前车所需的最小制动距离
        lead_0_obstacle = lead_xv_0[:,0] + get_stopped_equivalence_factor(lead_xv_0[:,1])
        lead_1_obstacle = lead_xv_1[:,0] + get_stopped_equivalence_factor(lead_xv_1[:,1])

        self.params[:,0] = ACCEL_MIN  # 最小加速度约束
        self.params[:,1] = ACCEL_MAX  # 最大加速度约束

        # ACC模式或混合模式更新
        if self.mode == 'acc':
            self.params[:,5] = LEAD_DANGER_FACTOR

            # 为巡航创建虚拟障碍物，确保在前车无影响时平滑加速到设定速度
            v_lower = v_ego + (T_IDXS * CRUISE_MIN_ACCEL * 1.05)  # 速度下限
            v_upper = v_ego + (T_IDXS * CRUISE_MAX_ACCEL * 1.05)  # 速度上限
            v_cruise_clipped = np.clip(v_cruise * np.ones(N+1), v_lower, v_upper)
            cruise_obstacle = np.cumsum(T_DIFFS * v_cruise_clipped) + get_safe_obstacle_distance(v_cruise_clipped, t_follow)
            x_obstacles = np.column_stack([lead_0_obstacle, lead_1_obstacle, cruise_obstacle])
            self.source = SOURCES[np.argmin(x_obstacles[0])]  # 选择最近的障碍物作为控制源

            # ACC模式下不使用这些参数
            x[:], v[:], a[:], j[:] = 0.0, 0.0, 0.0, 0.0

        elif self.mode == 'blended':
            self.params[:,5] = 1.0

            x_obstacles = np.column_stack([lead_0_obstacle, lead_1_obstacle])
            cruise_target = T_IDXS * np.clip(v_cruise, v_ego - 2.0, 1e3) + x[0]  # 巡航目标
            xforward = ((v[1:] + v[:-1]) / 2) * (T_IDXS[1:] - T_IDXS[:-1])  # 前向位置
            x = np.cumsum(np.insert(xforward, 0, x[0]))

            x_and_cruise = np.column_stack([x, cruise_target])
            x = np.min(x_and_cruise, axis=1)  # 取位置和巡航目标的最小值

            self.source = 'e2e' if x_and_cruise[1,0] < x_and_cruise[1,1] else 'cruise'

        else:
            raise NotImplementedError(f'Planner mode {self.mode} not recognized in planner update')

        # 设置参考轨迹
        self.yref[:,1] = x  # 位置参考
        self.yref[:,2] = v  # 速度参考
        self.yref[:,3] = a  # 加速度参考
        self.yref[:,5] = j  # 加加速度参考
        for i in range(N):
            self.solver.set(i, "yref", self.yref[i])
        self.solver.set(N, "yref", self.yref[N][:COST_E_DIM])

        # 设置参数
        self.params[:,2] = np.min(x_obstacles, axis=1)  # 最近障碍物距离
        self.params[:,3] = np.copy(self.prev_a)         # 上一时刻加速度
        self.params[:,4] = t_follow                     # 跟车时间

        self.run()  # 运行MPC求解

        # 碰撞检测
        if (np.any(lead_xv_0[FCW_IDXS,0] - self.x_sol[FCW_IDXS,0] < CRASH_DISTANCE) and
                radarstate.leadOne.modelProb > 0.9):
            self.crash_cnt += 1
        else:
            self.crash_cnt = 0

        # 检查是否进入前车舒适范围
        if self.mode == 'blended':
            if any((lead_0_obstacle - get_safe_obstacle_distance(self.x_sol[:,1], t_follow)) - self.x_sol[:,0] < 0.0):
                self.source = 'lead0'
            if any((lead_1_obstacle - get_safe_obstacle_distance(self.x_sol[:,1], t_follow)) - self.x_sol[:,0] < 0.0) and \
               (lead_1_obstacle[0] - lead_0_obstacle[0]):
                self.source = 'lead1'

    def run(self):
        """运行MPC求解器"""
        # 设置参数和约束
        for i in range(N+1):
            self.solver.set(i, 'p', self.params[i])
        self.solver.constraints_set(0, "lbx", self.x0)  # 状态下限
        self.solver.constraints_set(0, "ubx", self.x0)  # 状态上限

        # 求解MPC问题
        self.solution_status = self.solver.solve()

        # 记录求解时间
        self.solve_time = float(self.solver.get_stats('time_tot')[0])
        self.time_qp_solution = float(self.solver.get_stats('time_qp')[0])
        self.time_linearization = float(self.solver.get_stats('time_lin')[0])
        self.time_integrator = float(self.solver.get_stats('time_sim')[0])

        # 获取求解结果
        for i in range(N+1):
            self.x_sol[i] = self.solver.get(i, 'x')  # 状态解
        for i in range(N):
            self.u_sol[i] = self.solver.get(i, 'u')  # 控制解

        self.v_solution = self.x_sol[:,1]  # 速度解
        self.a_solution = self.x_sol[:,2]  # 加速度解
        self.j_solution = self.u_sol[:,0]  # 加加速度解

        # 更新上一时刻加速度
        self.prev_a = np.interp(T_IDXS + self.dt, T_IDXS, self.a_solution)

        # 处理求解失败情况
        t = time.monotonic()
        if self.solution_status != 0:
            if t > self.last_cloudlog_t + 5.0:  # 每5秒记录一次警告
                self.last_cloudlog_t = t
                cloudlog.warning(f"Long mpc reset, solution_status: {self.solution_status}")
            self.reset()  # 重置控制器


if __name__ == "__main__":
    """主函数：生成ACADOS求解器代码"""
    ocp = gen_long_ocp()
    AcadosOcpSolver.generate(ocp, json_file=JSON_FILE)
    # AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)