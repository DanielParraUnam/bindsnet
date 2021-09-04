from bindsnet.network.nodes import Nodes
import torch

class Burst(Nodes):
    
    def __init__(
        self,
        n = None,
        shape = None,
        traces: bool = False,
        tc_trace = 20.0,
        traces_additive: bool = False,
        trace_scale = 1.0,
        sum_input: bool = False,
        thresh = 80.0,
        reset = 15.0,
        resetvs = 30.0,
        resetvu = 20.0,
        lbound: float = None,
        learning: bool = True,
        **kwargs,
    ):
        super().__init__(
            n=n,
            shape=shape,
            traces=traces,
            traces_additive=traces_additive,
            tc_trace=tc_trace,
            trace_scale=trace_scale,
            sum_input=sum_input,
            learning=learning
        )

        self.register_buffer(
            "reset", torch.tensor(reset, dtype=torch.float)
        )  # Rest voltage.
        self.register_buffer(
            "resetvs", torch.tensor(resetvs, dtype=torch.float)
        )  # Rest voltage.
        self.register_buffer(
            "resetvu", torch.tensor(resetvu, dtype=torch.float)
        )  # Rest voltage.        
        self.register_buffer(
            "thresh", torch.tensor(thresh, dtype=torch.float)
        )  # Spike threshold voltage.
        self.register_buffer("v", torch.FloatTensor())  # Neuron voltages.
        self.register_buffer("vs", torch.FloatTensor())  # Neuron voltages.
        self.register_buffer("vu", torch.FloatTensor())  # Neuron voltages.

    def forward(self, x: torch.Tensor) -> None:
        """
        Runs a single simulation step.
        :param x: Inputs to the layer.
        """

        # Dynamics
        self.v += self.dt * (self.v**2 - 2*self.v*self.vs - self.vs**2 - self.vs - self.vu + 1.0 + x)
        self.vs += self.dt * (1/1.0) * (0.1 * self.v-self.vs).squeeze()
        self.vu += self.dt * (1/10.0) * (0.1 * self.v-self.vu).squeeze()

        # Check for spiking neurons.
        self.s = self.v >= self.thresh

        # Voltage reset.
        self.v.masked_fill_(self.s, self.reset)
        self.vs.masked_fill_(self.s, self.resetvs)
        self.vu += self.s * self.resetvu

        # Visualization bug... fix? 
        self.v = (self.v - (self.reset) + self.reset)
        # Probably tensor optimization... for some reason v tensor is not available while spike is
        # There is a reason why bindsnet has the class Monitor to track hidden voltages and spikes.

        super().forward(x)

    def reset_state_variables(self) -> None:
        """
        Resets relevant state variables.
        """
        super().reset_state_variables()
        self.v.fill_(self.reset)  
        self.vs.fill_(self.resetvs)
        self.vu += self.resetvu

    def set_batch_size(self, batch_size) -> None:
        """
        Sets mini-batch size. Called when layer is added to a network.
        :param batch_size: Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)
        self.v = torch.rand(batch_size, *self.shape, device=self.v.device) * 2 - 1
        self.vs = torch.rand(batch_size, *self.shape, device=self.v.device) * 2 - 1
        self.vu = torch.rand(batch_size, *self.shape, device=self.v.device) * 2 - 1
        

def main():
    import matplotlib.pyplot as plt


    n_neurons=100
    BurstLayer = Burst(
        n=n_neurons,
        reset=15.0,
        resetvs=30.0,
        resetvu=20.0,
        thresh=80.0,
        )

    BurstLayer.set_batch_size(1)
    BurstLayer.compute_decays(dt=0.001)

    Burstvoltages = []
    Burstspikes = []
    for t in range(50000):
        BurstLayer.forward(torch.zeros((1,n_neurons)))#torch.randint(0, 2, (1,n_neurons))) torch.ones((1,n_neurons))
        Burstvoltages.append(BurstLayer.v)
        Burstspikes.append(BurstLayer.s)

    plt.plot(torch.cat(Burstvoltages, dim=0)[:,torch.randperm(n_neurons)[25:30]])
    plt.show()

    plt.matshow(torch.cat(Burstspikes,dim=0).transpose(0,1))
    plt.show()

if __name__ == '__main__':
    main() 