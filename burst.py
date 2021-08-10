from bindsnet.network.nodes import Nodes

class Burst(Nodes):
    
    def __init__(
        self,
        n = None,
        shape = None,
        traces: bool = False,
        traces_additive: bool = False,
        tc_trace = 20.0,
        trace_scale = 1.0,
        sum_input: bool = False,
        thresh = -52.0,
        rest = 15.0,
        restvs = 30.0,
        restvu = 20.0,
        reset = -65.0,
        refrac = 5,
        tc_decay = 100.0,
        lbound: float = None,
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
        )

        self.register_buffer(
            "rest", torch.tensor(rest, dtype=torch.float)
        )  # Rest voltage.
        self.register_buffer(
            "restvs", torch.tensor(restvs, dtype=torch.float)
        )  # Rest voltage.
        self.register_buffer(
            "restvu", torch.tensor(restvu, dtype=torch.float)
        )  # Rest voltage.        
        self.register_buffer(
            "reset", torch.tensor(reset, dtype=torch.float)
        )  # Post-spike reset voltage.
        self.register_buffer(
            "thresh", torch.tensor(thresh, dtype=torch.float)
        )  # Spike threshold voltage.
        self.register_buffer(
            "refrac", torch.tensor(refrac)
        )  # Post-spike refractory period.
        self.register_buffer(
            "tc_decay", torch.tensor(tc_decay, dtype=torch.float)
        )  # Time constant of neuron voltage decay.
        self.register_buffer(
            "decay", torch.zeros(*self.shape)
        )  # Set in compute_decays.
        self.register_buffer("v", torch.FloatTensor())  # Neuron voltages.
        self.register_buffer("vs", torch.FloatTensor())  # Neuron voltages.
        self.register_buffer("vu", torch.FloatTensor())  # Neuron voltages.
        self.register_buffer(
            "refrac_count", torch.FloatTensor()
        )  # Refractory period counters.

        if lbound is None:
            self.lbound = None  # Lower bound of voltage.
        else:
            self.lbound = torch.tensor(
                lbound, dtype=torch.float
            )  # Lower bound of voltage.

    def forward(self, x: torch.Tensor) -> None:
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """
        # Decay voltages.
        self.v = self.decay * (self.v - self.rest) + self.rest
        # self.vs = self.decay * (self.vs - self.rest) + self.rest
        # self.vu = self.decay * (self.vu - self.rest) + self.rest
        
        # Integrate inputs.
        x.masked_fill_(self.refrac_count > 0, 0.0)

        # Decrement refractory counters.
        self.refrac_count -= self.dt

        # Dynamics
        self.v += self.dt * (self.v**2 - 2*self.v*self.vs - self.vs**2 - self.vs - self.vu + 1.0 + x)
        self.vs += self.dt * (1/1.0) * (0.1 * self.v-self.vs).squeeze()
        self.vu += self.dt * (1/10.0) * (0.1 * self.v-self.vu).squeeze()

        # Check for spiking neurons.
        self.s = self.v >= self.thresh

        # Refractoriness and voltage reset.
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, self.reset)

        # Voltage clipping to lower bound.
        if self.lbound is not None:
            self.v.masked_fill_(self.v < self.lbound, self.lbound)

        super().forward(x)

    def reset_state_variables(self) -> None:
        """
        Resets relevant state variables.
        """
        super().reset_state_variables()
        self.v = self.rest*torch.rand(self.v.shape, device=self.v.device) * 2 - 1
        self.vs = torch.rand(self.vs.shape, device=self.v.device) * 2 - 1
        self.vu = torch.rand(self.vu.shape, device=self.v.device) * 2 - 1
        
        #self.v.fill_(self.rest)  # Neuron voltages.
        #self.vs.fill_(self.restvs)
        #self.vu += self.restvu
        self.refrac_count.zero_()  # Refractory period counters.

    def compute_decays(self, dt) -> None:
        """
        Sets the relevant decays.
        """
        super().compute_decays(dt=dt)
        self.decay = torch.exp(
            -self.dt / self.tc_decay
        )  # Neuron voltage decay (per timestep).

    def set_batch_size(self, batch_size) -> None:
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)
        self.v = self.rest*torch.rand(batch_size, *self.shape, device=self.v.device) * 2 - 1
        self.vs = torch.rand(batch_size, *self.shape, device=self.v.device) * 2 - 1
        self.vu = torch.rand(batch_size, *self.shape, device=self.v.device) * 2 - 1
        self.refrac_count = torch.zeros_like(self.v, device=self.refrac_count.device)


def main():
	BurstLayer = Burst(
            n=n_neurons,
            traces=True,
            rest=15.0,
            restvs=30.0,
            restvu=20.0,
            reset=15.0,
            thresh=80.0,
            tc_decay=10.0,
            refrac=0,
            tc_trace=20.0,
        )
	BurstLayer.set_batch_size(1)
	BurstLayer.compute_decays(0.1)

	Burstvoltages = []
	Burstspikes = []
	for t in range(250):
	    BurstLayer.forward(torch.zeros((1,n_neurons)))#torch.randint(0, 2, (1,n_neurons))) torch.ones((1,n_neurons))
	    Burstvoltages.append(BurstLayer.v)
	    Burstspikes.append(BurstLayer.s)

	plt.plot(torch.cat(Burstvoltages, dim=0)[:,torch.randperm(n_neurons)[25:30]])
	plt.show()

	plt.matshow(torch.cat(Burstspikes,dim=0).transpose(0,1))

if __name__ == '__main__':
    main()        