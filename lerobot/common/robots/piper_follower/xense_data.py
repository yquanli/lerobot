from xensesdk import ExampleView, Sensor
import sys



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

        self.view = ExampleView(self.sensor)
        self.view2d = self.view.create2d(Sensor.OutputType.Difference, Sensor.OutputType.Depth,Sensor.OutputType.Marker2D)

        self.view.setCallback(self._callback)

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
        try:
            self.view.show()
        finally:
            self.sensor.release()
            sys.exit()


if __name__ == '__main__':
    xense = Xense()
    xense.run()
