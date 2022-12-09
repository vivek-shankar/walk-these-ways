/*****************************************************************
 Copyright (c) 2020, Unitree Robotics.Co.Ltd. All rights reserved.
******************************************************************/

#include "unitree_legged_sdk/unitree_legged_sdk.h"
#include "unitree_legged_sdk/unitree_joystick.h"
#include <math.h>
#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <thread>
#include <lcm/lcm-cpp.hpp>
#include "state_estimator_lcmt.hpp"
#include "leg_control_data_lcmt.hpp"
#include "pd_tau_targets_lcmt.hpp"
#include "rc_command_lcmt.hpp"

using namespace std;
using namespace UNITREE_LEGGED_SDK;

class Custom
{
public:
    Custom(uint8_t level): safe(LeggedType::A1), udp(level) {
        udp.InitCmdData(cmd);
    }
    void UDPRecv();
    void UDPSend();
    void RobotControl();
    void init();
    void handleActionLCM(const lcm::ReceiveBuffer *rbuf, const std::string & chan, const pd_tau_targets_lcmt * msg);
    void _simpleLCMThread();

    Safety safe;
    UDP udp;
    LowCmd cmd = {0};
    LowState state = {0};
    int motiontime = 0;
    float dt = 0.002;     // 0.001~0.01

    lcm::LCM _simpleLCM;
    std::thread _simple_LCM_thread;
    bool _firstCommandReceived;
    bool _firstRun;
    state_estimator_lcmt body_state_simple = {0};
    leg_control_data_lcmt joint_state_simple = {0};
    pd_tau_targets_lcmt joint_command_simple = {0};
    rc_command_lcmt rc_command = {0};

    xRockerBtnDataStruct _keyData;
    int mode = 0;
    float init_q[12] = {-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 0.8, -1.5};

};

void Custom::init()
{
    _simpleLCM.subscribe("pd_plustau_targets", &Custom::handleActionLCM, this);
    _simple_LCM_thread = std::thread(&Custom::_simpleLCMThread, this);

    _firstCommandReceived = false;
    _firstRun = true;

    // set nominal pose
    for(int i = 0; i < 12; i++){
        joint_command_simple.qd_des[i] = 0;
        joint_command_simple.tau_ff[i] = 0;
        joint_command_simple.kp[i] = 20.;
        joint_command_simple.kd[i] = 0.5;
    }

    joint_command_simple.q_des[0] = init_q[0];
    joint_command_simple.q_des[1] = init_q[1];
    joint_command_simple.q_des[2] = init_q[2];
    joint_command_simple.q_des[3] = init_q[3];
    joint_command_simple.q_des[4] = init_q[4];
    joint_command_simple.q_des[5] = init_q[5];
    joint_command_simple.q_des[6] = init_q[6];
    joint_command_simple.q_des[7] = init_q[7];
    joint_command_simple.q_des[8] = init_q[8];
    joint_command_simple.q_des[9] = init_q[9];
    joint_command_simple.q_des[10] = init_q[10];
    joint_command_simple.q_des[11] = init_q[11];

    printf("SET NOMINAL POSE");
}

void Custom::UDPRecv()
{
    udp.Recv();
}

void Custom::UDPSend()
{
    udp.Send();
}

double jointLinearInterpolation(double initPos, double targetPos, double rate)
{
    double p;
    rate = std::min(std::max(rate, 0.0), 1.0);
    p = initPos*(1-rate) + targetPos*rate;
    return p;
}

void Custom::handleActionLCM(const lcm::ReceiveBuffer *rbuf, const std::string & chan, const pd_tau_targets_lcmt * msg){
    (void) rbuf;
    (void) chan;

    joint_command_simple = *msg;
    _firstCommandReceived = true;

}

void Custom::_simpleLCMThread(){
    while(true){
        _simpleLCM.handle();
    }
}

void Custom::RobotControl()
{
    motiontime++;
    udp.GetRecv(state);
    memcpy(&_keyData, state.wirelessRemote, 40);

    rc_command.left_stick[0] = _keyData.lx;
    rc_command.left_stick[1] = _keyData.ly;
    rc_command.right_stick[0] = _keyData.rx;
    rc_command.right_stick[1] = _keyData.ry;
    rc_command.right_lower_right_switch = _keyData.btn.components.R2;
    rc_command.right_upper_switch = _keyData.btn.components.R1;
    rc_command.left_lower_left_switch = _keyData.btn.components.L2;
    rc_command.left_upper_switch = _keyData.btn.components.L1;


    if(_keyData.btn.components.A > 0){
        mode = 0;
    } else if(_keyData.btn.components.B > 0){
        mode = 1;
    }else if(_keyData.btn.components.X > 0){
        mode = 2;
    }else if(_keyData.btn.components.Y > 0){
        mode = 3;
    }else if(_keyData.btn.components.up > 0){
        mode = 4;
    }else if(_keyData.btn.components.right > 0){
        mode = 5;
    }else if(_keyData.btn.components.down > 0){
        mode = 6;
    }else if(_keyData.btn.components.left > 0){
        mode = 7;
    }

    rc_command.mode = mode;


    // publish state to LCM
    for(int i = 0; i < 12; i++){
        joint_state_simple.q[i] = state.motorState[i].q;
        joint_state_simple.qd[i] = state.motorState[i].dq;
    }
    for(int i = 0; i < 4; i++){
        body_state_simple.quat[i] = state.imu.quaternion[i];
    }
    for(int i = 0; i < 3; i++){
        body_state_simple.rpy[i] = state.imu.rpy[i];
        body_state_simple.aBody[i] = state.imu.accelerometer[i];
        body_state_simple.omegaBody[i] = state.imu.gyroscope[i];
    }
    for(int i = 0; i < 4; i++){
        body_state_simple.contact_estimate[i] = state.footForce[i];
    }

    // std::cout <<"Current: " << joint_state_simple.q[0] << " " << joint_state_simple.q[1] << " " << joint_state_simple.q[2] << " " 
    //                         << joint_state_simple.q[3] << " " << joint_state_simple.q[4] << " " << joint_state_simple.q[5] << " " 
    //                         << joint_state_simple.q[6] << " " << joint_state_simple.q[7] << " " << joint_state_simple.q[8] << " " 
    //                         << joint_state_simple.q[9] << " " << joint_state_simple.q[10] << " " << joint_state_simple.q[11] << std::endl;

    _simpleLCM.publish("state_estimator_data", &body_state_simple);
    _simpleLCM.publish("leg_control_data", &joint_state_simple);
    _simpleLCM.publish("rc_command", &rc_command);

    if(_firstRun && joint_state_simple.q[0] != 0){
        for(int i = 0; i < 12; i++){
            joint_command_simple.q_des[i] = joint_state_simple.q[i];
        }
        _firstRun = false;
    }

    for(int i = 0; i < 12; i++){
        cmd.motorCmd[i].q = joint_command_simple.q_des[i];
        cmd.motorCmd[i].dq = joint_command_simple.qd_des[i];
        cmd.motorCmd[i].Kp = joint_command_simple.kp[i];
        cmd.motorCmd[i].Kd = joint_command_simple.kd[i];
        cmd.motorCmd[i].tau = joint_command_simple.tau_ff[i];
    }

    safe.PositionLimit(cmd);
    safe.PowerProtect(cmd, state, 9);
    // std::cout <<"Command: " << cmd.motorCmd[0].q << " " << cmd.motorCmd[1].q << " " << cmd.motorCmd[2].q << " " 
    //                         << cmd.motorCmd[3].q << " " << cmd.motorCmd[4].q << " " << cmd.motorCmd[5].q << " " 
    //                         << cmd.motorCmd[6].q << " " << cmd.motorCmd[7].q << " " << cmd.motorCmd[8].q << " " 
    //                         << cmd.motorCmd[9].q << " " << cmd.motorCmd[10].q << " " << cmd.motorCmd[11].q << std::endl; 
    udp.SetSend(cmd);
}


int main(void)
{
    std::cout << "Communication level is set to LOW-level." << std::endl
              << "WARNING: Make sure the robot is hung up." << std::endl
              << "Press Enter to continue..." << std::endl;
    std::cin.ignore();

    Custom custom(LOWLEVEL);
    custom.init();
    
    LoopFunc loop_control("control_loop", custom.dt,    boost::bind(&Custom::RobotControl, &custom));
    LoopFunc loop_udpSend("udp_send",     custom.dt, 3, boost::bind(&Custom::UDPSend,      &custom));
    LoopFunc loop_udpRecv("udp_recv",     custom.dt, 3, boost::bind(&Custom::UDPRecv,      &custom));

    loop_udpSend.start();
    loop_udpRecv.start();
    loop_control.start();

    while(1){
        sleep(10);
    };

    return 0;
}
