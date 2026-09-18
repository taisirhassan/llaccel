# Optional diagnostic profile: bound timing-repair runtime without changing SDC.
# Installed via ORFS PRE_*_TCL hooks; the default flow does not source this file.
if {![info exists ::env(LLACCEL_REPAIR_MAX_ITERATIONS)] ||
    ![string is integer -strict $::env(LLACCEL_REPAIR_MAX_ITERATIONS)] ||
    $::env(LLACCEL_REPAIR_MAX_ITERATIONS) <= 0} {
  error "LLACCEL_REPAIR_MAX_ITERATIONS must be a positive integer"
}
if {![llength [info commands ::llaccel_original_repair_timing]]} {
  rename ::repair_timing ::llaccel_original_repair_timing
  proc ::repair_timing {args} {
    # Preserve an explicitly stricter caller limit if one exists.
    set at [lsearch -exact $args -max_iterations]
    set bound $::env(LLACCEL_REPAIR_MAX_ITERATIONS)
    if {$at >= 0} {
      set existing [lindex $args [expr {$at + 1}]]
      if {[string is integer -strict $existing] && $existing > 0} {
        set bound [expr {min($existing, $bound)}]
      }
      set args [lreplace $args $at [expr {$at + 1}]]
    }
    lappend args -max_iterations $bound
    puts "LLACCEL bounded diagnostic repair: max_iterations=$bound; clock constraints unchanged"
    tailcall ::llaccel_original_repair_timing {*}$args
  }
}
