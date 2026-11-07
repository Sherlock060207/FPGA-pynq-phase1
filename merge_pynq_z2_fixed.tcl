# ============================================================
# Merge conv3x3 HLS IP into the PYNQ-Z2 base design
# Fixed for Vivado 2026.1 — all logic on FCLK_CLK0 (100 MHz)
# ============================================================

set script_dir [file dirname [info script]]
set base_dir  [file normalize [file join $script_dir pynq-2.3-source boards Pynq-Z2 base]]
set ip_repo   [file normalize [file join $script_dir pynq_conv3x3_hls conv3x3_hlssolution1 hls impl ip]]
set out_dir   [file normalize [file join $script_dir vision_accel]]
file mkdir $out_dir

# --- Step 1: build unmodified PYNQ-Z2 base project ---
cd $out_dir
source [file join $base_dir base_patched.tcl]
open_project  [file join $out_dir base base.xpr]
open_bd_design [file join $out_dir base base.srcs sources_1 bd base base.bd]
set_property ip_repo_paths $ip_repo [current_project]
update_ip_catalog

# --- Step 2: enable S_AXI_HP1 on PS (base design leaves it disabled) ---
# HP0 is taken by VDMA, HP2 by trace_analyzer DMA; HP1 is free.
set_property -dict [list CONFIG.PCW_USE_S_AXI_HP1 {1}] [get_bd_cells ps7_0]

# --- Step 3: add HLS accelerator cell ---
set conv [create_bd_cell -type ip -vlnv xilinx.com:hls:conv3x3:1.0 conv3x3_0]

# --- Step 4: control path -----------------------------------
# GP0 interconnect (ps7_0_axi_periph, 10 MI ports, FCLK_CLK0) already
# has M08 free.  Hook s_axi_control directly — no extra interconnect.
connect_bd_intf_net [get_bd_intf_pins ps7_0_axi_periph/M08_AXI] \
                    [get_bd_intf_pins conv3x3_0/s_axi_control]

# --- Step 5: data path --------------------------------------
# gmem0 + gmem1 are both AXI masters on the same clock (ap_clk = FCLK_CLK0).
# Merge them through a 2-slave / 1-master interconnect into HP1.
set data_ic [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 conv_data_ic]
set_property -dict [list CONFIG.NUM_MI {1} CONFIG.NUM_SI {2}] $data_ic
connect_bd_intf_net [get_bd_intf_pins conv3x3_0/m_axi_gmem0] [get_bd_intf_pins conv_data_ic/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins conv3x3_0/m_axi_gmem1] [get_bd_intf_pins conv_data_ic/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins conv_data_ic/M00_AXI]  [get_bd_intf_pins ps7_0/S_AXI_HP1]

# --- Step 6: clock (FCLK_CLK0 = 100 MHz, single clock domain) ---
connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0] \
  [get_bd_pins conv3x3_0/ap_clk] \
  [get_bd_pins conv_data_ic/ACLK] \
  [get_bd_pins conv_data_ic/S00_ACLK] \
  [get_bd_pins conv_data_ic/S01_ACLK] \
  [get_bd_pins conv_data_ic/M00_ACLK] \
  [get_bd_pins ps7_0/S_AXI_HP1_ACLK]

# --- Step 7: reset (fclk0 domain) ---
# ARESETN (interconnect global) -> interconnect_aresetn
# Sxx_ARESETN / Mxx_ARESETN (per-interface) -> peripheral_aresetn
connect_bd_net [get_bd_pins rst_ps7_0_fclk0/interconnect_aresetn] \
  [get_bd_pins conv_data_ic/ARESETN]
connect_bd_net [get_bd_pins rst_ps7_0_fclk0/peripheral_aresetn] \
  [get_bd_pins conv3x3_0/ap_rst_n] \
  [get_bd_pins conv_data_ic/S00_ARESETN] \
  [get_bd_pins conv_data_ic/S01_ARESETN] \
  [get_bd_pins conv_data_ic/M00_ARESETN]

# --- Step 8: address, validate, save, build ---
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
puts "=============================================="
puts " Generated vision_accel.bit and vision_accel.hwh"
puts " Output dir: $out_dir"
puts "=============================================="
