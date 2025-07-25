from xensesdk import ExampleView, Sensor
import sys
import threading


class Xense:
    def __init__(self, device_id="OG000205"):
        self.device_id = device_id
        self.sensor = Sensor.create(self.device_id)

        self.latest_diff = None
        self.latest_rectify = None
        self.latest_depth = None
        self.latest_force = None
        self.latest_force_norm = None
        self.latest_force_res = None
        self.latest_mesh_init = None
        self.latest_mesh_now = None
        self.latest_mesh_flow = None

        # self.view = ExampleView(self.sensor)
        # self.view2d = self.view.create2d(Sensor.OutputType.Difference, Sensor.OutputType.Depth,Sensor.OutputType.Marker2D)
        # self.view.setCallback(self._callback)

    def _callback(self):
        diff, rectify, depth, force, force_norm, force_res, mesh_init, mesh_now, mesh_flow = self.sensor.selectSensorInfo(
            Sensor.OutputType.Difference,
            Sensor.OutputType.Rectify,
            Sensor.OutputType.Depth,
            Sensor.OutputType.Force,
            Sensor.OutputType.ForceNorm,
            Sensor.OutputType.ForceResultant,
            Sensor.OutputType.Mesh3DInit,
            Sensor.OutputType.Mesh3D,
            Sensor.OutputType.Mesh3DFlow
        )

        self.latest_diff = diff
        self.latest_rectify = rectify
        self.latest_depth = depth
        self.latest_force = force
        self.latest_force_norm = force_norm
        self.latest_force_res = force_res
        self.latest_mesh_init = mesh_init
        self.latest_mesh_now = mesh_now
        self.latest_mesh_flow = mesh_flow

        
        marker_img = self.sensor.drawMarkerMove(self.latest_rectify)
        self.view2d.setData(Sensor.OutputType.Marker2D, marker_img)
        self.view2d.setData(Sensor.OutputType.Difference, diff)
        self.view2d.setData(Sensor.OutputType.Depth, depth)
        self.view.setForceFlow(force, force_res, mesh_init)
        self.view.setDepth(depth)

    def read_data(self):
        
        return (
            self.latest_diff,
            self.latest_rectify,
            self.latest_depth,
            self.latest_force,
            self.latest_force_norm,
            self.latest_force_res,
            self.latest_mesh_init,
            self.latest_mesh_now,
            self.latest_mesh_flow
        )

    def run(self):
            diff, rectify, depth, force, force_norm, force_res, mesh_init, mesh_now, mesh_flow = self.sensor.selectSensorInfo(
            Sensor.OutputType.Difference,
            Sensor.OutputType.Rectify,
            Sensor.OutputType.Depth,
            Sensor.OutputType.Force,
            Sensor.OutputType.ForceNorm,
            Sensor.OutputType.ForceResultant,
            Sensor.OutputType.Mesh3DInit,
            Sensor.OutputType.Mesh3D,
            Sensor.OutputType.Mesh3DFlow
        )
            self.latest_diff = diff
            self.latest_rectify = rectify
            self.latest_depth = depth
            self.latest_force = force
            self.latest_force_norm = force_norm
            self.latest_force_res = force_res
            self.latest_mesh_init = mesh_init
            self.latest_mesh_now = mesh_now
            self.latest_mesh_flow = mesh_flow
        
            

    
if __name__ == '__main__':
    
    xense_0=Xense()
    xense_1=Xense(device_id="OG000203")
    while(True):
        xense_0.run()
        xense_1.run()
        print(xense_0.read_data())
        print(xense_1.read_data())
    #     (
    #     xense_0_diff,
    #     xense_0_rectify,
    #     xense_0_depth,
    #     xense_0_force,
    #     xense_0_force_norm,
    #     xense_0_force_resultant,
    #     xense_0_mesh_init,
    #     xense_0_mesh_now,
    #     xense_0_mesh_flow
    #   ) = xense_0.read_data()
    
    #     (
    #     xense_1_diff,
    #     xense_1_rectify,
    #     xense_1_depth,
    #     xense_1_force,
    #     xense_1_force_norm,
    #     xense_1_force_resultant,
    #     xense_1_mesh_init,
    #     xense_1_mesh_now,
    #     xense_1_mesh_flow
    #   ) = xense_1.read_data()

#    obs_dict["xense_0_rectify"] = xense_0_rectify  # (700, 400, 3)
# obs_dict["xense_0_diff"] = xense_0_diff  # (700, 400, 3)
# obs_dict["xense_0_depth"] = xense_0_depth  # (700, 400, 1)
# obs_dict["xense_0_force"] = xense_0_force  # (35, 20, 3)
#                 obs_dict["xense_0_force_norm"] = xense_0_force_norm  # (35, 20, 3)
#                 obs_dict["xense_0_force_resultant"] = xense_0_force_resultant  # (6,)
#                 obs_dict["xense_0_mesh_init"] = xense_0_mesh_init  # (35, 20, 3)
#                 obs_dict["xense_0_mesh_now"] = xense_0_mesh_now    # (35, 20, 3)
#                 obs_dict["xense_0_mesh_flow"] = xense_0_mesh_flow  # (35, 20, 3)

#                 obs_dict["xense_1_rectify"] = xense_1_rectify  # (700, 400, 3)
#                 obs_dict["xense_1_diff"] = xense_1_diff  # (700, 400, 3)
#                 obs_dict["xense_1_depth"] = xense_1_depth  # (700, 400, 1)
#                 obs_dict["xense_1_force"] = xense_1_force  # (35, 20, 3)
#                 obs_dict["xense_1_force_norm"] = xense_1_force_norm  # (35, 20, 3)
#                 obs_dict["xense_1_force_resultant"] = xense_1_force_resultant  # (6,)
#                 obs_dict["xense_1_mesh_init"] = xense_1_mesh_init  # (35, 20, 3)
#                 obs_dict["xense_1_mesh_now"] = xense_1_mesh_now    # (35, 20, 3)
#                 obs_dict["xense_1_mesh_flow"] = xense_1_mesh_flow  # (35, 20, 3)


    #     with open("observation.txt", "w") as f:
    #         for key, value in obs_dict.items():
    #              f.write(f"{key}: {np.array(value).tolist()}\n")