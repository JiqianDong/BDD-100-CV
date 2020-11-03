Thanks for using our dataset!

We are using one-hot encoding to represent the annotations. 
Specifically, for the action, the labels are ['move forward', 'stop/slow down', 'turn left', 'turn right'].

For the explanation, the labels are: [
f_follow_traffic: Forward - follow traffic,
f_road_clear: Forward - the road is clear, 
f_traffic_light: Forward - the traffic light is green, 
s_ob_car: Stop/slow down - obstacle: car, 
s_ob_ped: Stop/slow down - obstacle: person/pedestrain, 
s_ob_rider: Stop/slow down - obstacle: rider, 
s_other: Stop/slow down - obstacle: others, 
s_traffic_light: Stop/slow down - the traffic light, 
s_traffic_sign: Stop/slow down - the traffic sign, 
l_front_car: Turn left - front car turning left, 
l_lane: Turn left - on the left-turn lane, 
l_traffic_light: Turn left - traffic light allows, 
r_front_car: Turn right - front car turning right, 
r_lane: Turn right - on the right-turn lane, 
r_traffic_light: Turn right - traffic light allows, 
no_l_car - Can't turn left - obstacles on the left lane, 
no_l_lane - Can't turn left - no lane on the left, 
no_l_solid_line - Can't turn left - solid line on the left, 
no_r_car - Can't turn right - obstacles on the right lane, 
no_r_lane - Can't turn right - no lane on the right, 
no_r_solid_line - Can't turn right - solid line on the right,
].
