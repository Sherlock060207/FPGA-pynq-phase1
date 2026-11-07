# Merge conv3x3 HLS IP into the PYNQ-Z2 base design
set base_dir [file normalize [file join [file dirname [info script]] .. pynq-2.3-source boards Pynq-Z2 base]]
set ip_repo [file normalize [file join [file dirname [info script]] pynq_conv3x3_hls conv3x3_hlssolution1 hls impl ip]]
set out_dir [file normalize [file join [file dirname [info script]] vision_accel]]
file mkdir $out_dir

# Build the unmodified PYNQ-Z2 base project first
cd $out_dir
source [file join $base_dir base.tcl]
open_project [file join $out_dir base base.xpr]
open_bd_design [file join $out_dir base base.srcs sources_1 bd base base.bd]
set_property ip_repo_paths $ip_repo [current_project]
update_ip_catalog

# Add HLS accelerator
set conv [create_bd_cell -type ip -vlnv xilinx.com:hls:conv3x3:1.0 conv3x3_0]

# Control path: PS GP1 -> small AXI interconnect -> accelerator AXI-Lite
set ctrl_ic [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 conv_ctrl_ic]
set_property -dict [list CONFIG.NUM_MI {1} CONFIG.NUM_SI {1}] $ctrl_ic
connect_bd_intf_net [get_bd_intf_pins ps7_0/M_AXI_GP1] [get_bd_intf_pins conv_ctrl_ic/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins conv_ctrl_ic/M00_AXI] [get_bd_intf_pins conv3x3_0/s_axi_control]

# Data path: accelerator gmem ports -> HP0 DDR through a 2-slave interconnect
set data_ic [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 conv_data_ic]
set_property -dict [list CONFIG.NUM_MI {1} CONFIG.NUM_SI {2}] $data_ic
connect_bd_intf_net [get_bd_intf_pins conv3x3_0/m_axi_gmem0] [get_bd_intf_pins conv_data_ic/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins conv3x3_0/m_axi_gmem1] [get_bd_intf_pins conv_data_ic/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins conv_data_ic/M00_AXI] [get_bd_intf_pins ps7_0/S_AXI_HP0]

# Clock and reset
connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0] [get_bd_pins conv3x3_0/ap_clk] [get_bd_pins conv_ctrl_ic/ACLK] [get_bd_pins conv_ctrl_ic/S00_ACLK] [get_bd_pins conv_ctrl_ic/M00_ACLK]
connect_bd_net [get_bd_pins ps7_0/FCLK_CLK1] [get_bd_pins conv_data_ic/ACLK] [get_bd_pins conv_data_ic/S00_ACLK] [get_bd_pins conv_data_ic/S01_ACLK] [get_bd_pins conv_data_ic/M00_ACLK]
connect_bd_net [get_bd_pins rst_ps7_0_fclk0/peripheral_aresetn] [get_bd_pins conv3x3_0/ap_rst_n] [get_bd_pins conv_ctrl_ic/ARESETN] [get_bd_pins conv_ctrl_ic/S00_ARESETN] [get_bd_pins conv_ctrl_ic/M00_ARESETN]
connect_bd_net [get_bd_pins rst_ps7_0_fclk1/peripheral_aresetn] [get_bd_pins conv_data_ic/ARESETN] [get_bd_pins conv_data_ic/S00_ARESETN] [get_bd_pins conv_data_ic/S01_ARESETN] [get_bd_pins conv_data_ic/M00_ARESETN]

assign_bd_address
validate_bd_design
save_bd_design
make_wrapper -files [get_files [file join $out_dir base base.srcs sources_1 bd base base.bd]] -top
add_files -norecurse [file join $out_dir base base.srcs sources_1 bd base hdl base_wrapper.v]
set_property top base_wrapper [current_fileset]
update_compile_order -fileset sources_1
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
file copy -force [file join $out_dir base base.runs impl_1 base_wrapper.bit] [file join $out_dir vision_accel.bit]
write_hwdef -force -file [file join $out_dir vision_accel.hwh]
puts "Generated vision_accel.bit and vision_accel.hwh"
