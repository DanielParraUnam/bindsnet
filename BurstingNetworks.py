from bindsnet.network.network import Network
import burst

class DiehlAndCookBursting(Network):
    """
    Implements the spiking neural network architecture from `(Diehl & Cook 2015)
    <https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full>`_.
    """

    def __init__(
        self,
        n_inpt: int,
        n_neurons: int = 100,
        exc: float = 22.5,
        inh: float = 17.5,
        dt: float = 1.0,
        nu = (1e-4, 1e-2),
        reduction = None,
        wmin: float = 0.0,
        wmax: float = 1.0,
        norm: float = 78.4,
        theta_plus: float = 0.05,
        tc_theta_decay: float = 1e7,
        inpt_shape = None,
    ) -> None:
        # language=rst
        """
        Constructor for class ``DiehlAndCook2015``.

        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of excitatory, inhibitory neurons.
        :param exc: Strength of synapse weights from excitatory to inhibitory layer.
        :param inh: Strength of synapse weights from inhibitory to excitatory layer.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param wmin: Minimum allowed weight on input to excitatory synapses.
        :param wmax: Maximum allowed weight on input to excitatory synapses.
        :param norm: Input to excitatory layer connection weights normalization
            constant.
        :param theta_plus: On-spike increment of ``DiehlAndCookNodes`` membrane
            threshold potential.
        :param tc_theta_decay: Time constant of ``DiehlAndCookNodes`` threshold
            potential decay.
        :param inpt_shape: The dimensionality of the input layer.
        """
        super().__init__(dt=dt)

        self.n_inpt = n_inpt
        self.inpt_shape = inpt_shape
        self.n_neurons = n_neurons
        self.exc = exc
        self.inh = inh
        self.dt = dt

        # Layers
        input_layer = Input(
            n=self.n_inpt, shape=self.inpt_shape, traces=True, tc_trace=20.0
        )
        exc_layer = burst.Burst(
            n=self.n_neurons,
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
        
        inh_layer = LIFNodes(
            n=self.n_neurons,
            traces=False,
            rest=-60.0,
            reset=-45.0,
            thresh=-40.0,
            tc_decay=10.0,
            refrac=2,
            tc_trace=20.0,
        )

        # Connections
        w = 0.3 * torch.rand(self.n_inpt, self.n_neurons)/100
        input_exc_conn = Connection(
            source=input_layer,
            target=exc_layer,
            w=w,
            update_rule=PostPre,
            nu=nu,
            reduction=reduction,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
        )
        w = self.exc * torch.diag(torch.ones(self.n_neurons))
        exc_inh_conn = Connection(
            source=exc_layer, target=inh_layer, w=w, wmin=0, wmax=self.exc
        )
        w = -self.inh * (
            torch.ones(self.n_neurons, self.n_neurons)
            - torch.diag(torch.ones(self.n_neurons))
        )
        inh_exc_conn = Connection(
            source=inh_layer, target=exc_layer, w=w, wmin=-self.inh, wmax=0
        )

        # Add to network
        self.add_layer(input_layer, name="X")
        self.add_layer(exc_layer, name="Ae")
        self.add_layer(inh_layer, name="Ai")
        self.add_connection(input_exc_conn, source="X", target="Ae")
        self.add_connection(exc_inh_conn, source="Ae", target="Ai")
        self.add_connection(inh_exc_conn, source="Ai", target="Ae")

class BurstingFC(Network):
    """
    Implements the spiking neural network architecture from `(Diehl & Cook 2015)
    <https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full>`_.
    """

    def __init__(
        self,
        n_inpt: int,
        n_neurons: int = 100,
        exc: float = 22.5,
        inh: float = 17.5,
        dt: float = 1.0,
        nu = (1e-4, 1e-2),
        reduction = None,
        wmin: float = 0.0,
        wmax: float = 1.0,
        norm: float = 78.4,
        theta_plus: float = 0.05,
        tc_theta_decay: float = 1e7,
        inpt_shape = None,
    ) -> None:

class BurstingConv(Network):
    """
    Implements the spiking neural network architecture from `(Diehl & Cook 2015)
    <https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full>`_.
    """

    def __init__(
        self,
        n_inpt: int,
        n_neurons: int = 100,
        exc: float = 22.5,
        inh: float = 17.5,
        dt: float = 1.0,
        nu = (1e-4, 1e-2),
        reduction = None,
        wmin: float = 0.0,
        wmax: float = 1.0,
        norm: float = 78.4,
        theta_plus: float = 0.05,
        tc_theta_decay: float = 1e7,
        inpt_shape = None,
    ) -> None:


def main():
    dt=0.1
    time=50
    network = DiehlAndCookBursting(
        n_inpt=784,
        n_neurons=n_neurons,
        exc=exc,
        inh=inh,
        dt=0.1,
        norm=78.4,
        theta_plus=theta_plus,
        inpt_shape=(1, 28, 28),
    )

    # Load MNIST data.
    train_dataset = MNIST(
        PoissonEncoder(time=time, dt=dt),
        None,
        root=os.path.join("..", "..", "data", "MNIST"),
        download=True,
        train=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
        ),
    )

    # Record spikes during the simulation.
    spike_record = torch.zeros((1, int(time / dt), n_neurons), device=device)

    # Neuron assignments and spike proportions.
    n_classes = 10
    assignments = -torch.ones(n_neurons, device=device)
    proportions = torch.zeros((n_neurons, n_classes), device=device)
    rates = torch.zeros((n_neurons, n_classes), device=device)

    # Voltage recording for excitatory and inhibitory layers.
    exc_voltage_monitor = Monitor(
        network.layers["Ae"], ["v"], time=int(time / dt), device=device
    )
    inh_voltage_monitor = Monitor(
        network.layers["Ai"], ["v"], time=int(time / dt), device=device
    )
    network.add_monitor(exc_voltage_monitor, name="exc_voltage")
    network.add_monitor(inh_voltage_monitor, name="inh_voltage")

    # Set up monitors for spikes and voltages
    spikes = {}
    for layer in set(network.layers):
        spikes[layer] = Monitor(
            network.layers[layer], state_vars=["s"], time=int(time / dt), device=device
        )
        network.add_monitor(spikes[layer], name="%s_spikes" % layer)

    voltages = {}
    for layer in set(network.layers) - {"X"}:
        voltages[layer] = Monitor(
            network.layers[layer], state_vars=["v"], time=int(time / dt), device=device
        )
        
    # Create a dataloader to iterate and batch data
    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=n_workers, pin_memory=gpu
    )

    #network.train(mode=False)

    # Get next input sample.
    single_image = next(iter(dataloader))
    inputs = {"X": single_image["encoded_image"].view(int(time / dt), 1, 1, 28, 28)}
    if gpu:
        inputs = {k: v.cuda() for k, v in inputs.items()}
        
    # Run the network on the input.
    network.run(inputs=inputs, time=int(time), input_time_dim=1)

    # Get voltage recording.
    exc_voltages = exc_voltage_monitor.get("v")
    inh_voltages = inh_voltage_monitor.get("v")
    spikes_ = {layer: spikes[layer].get("s") for layer in spikes}
    voltages = {"Ae": exc_voltages, "Ai": inh_voltages}

    inpt_ims, inpt_axes = None, None
    spike_ims, spike_axes = None, None
    weights_im = None
    assigns_im = None
    perf_ax = None
    voltage_axes, voltage_ims = None, None


    # Optionally plot various simulation information.
    image = single_image["image"].view(28, 28)
    inpt = inputs["X"].view(int(time/dt), 784).sum(0).view(28, 28)
    input_exc_weights = network.connections[("X", "Ae")].w
    square_weights = get_square_weights(
        input_exc_weights.view(784, n_neurons), n_sqrt, 28
    )
    square_assignments = get_square_assignments(assignments, n_sqrt)

    inpt_axes, inpt_ims = plot_input(
        image, inpt, label=single_image["label"], axes=inpt_axes, ims=inpt_ims
    )
    spike_ims, spike_axes = plot_spikes(spikes_, ims=spike_ims, axes=spike_axes)
    weights_im = plot_weights(square_weights, im=weights_im)
    #assigns_im = plot_assignments(square_assignments, im=assigns_im)
    voltage_ims, voltage_axes = plot_voltages(
        voltages, ims=voltage_ims, axes=voltage_axes, plot_type="line"
    )

    plt.pause(1e-8)

    network.reset_state_variables()  # Reset state variables.

if __name__ == '__main__':
    main()        