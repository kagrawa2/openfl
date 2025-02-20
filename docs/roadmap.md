# Roadmap

This document is intended to give users and contributors an idea of the OpenFL team's current priorities, features we plan to incorporate over the short, medium, and long term, and call out opportunities for the community to get involved.

### When will this document be updated?
We expect to update this document at least once every quarter.

## Long-term directions

### Decoupling the FL specification interface from the infrastructure
The task runner interface is coupled with the the single experiment aggregator / collaborator infrastructure, and the Interactive API is tied to the director / envoy infrastructure. 
The Interactive API was originally designed to be a high-level API for OpenFL, but for the cases when more control is required by users, access to lower level interfaces is necessary.
In OpenFL 1.5, we introduced the Workflow API as an experimental feature, which can be used to specify the federated learning flow, independently of the underlying computing infrastructure. The Workflow API facilitates a seamless transition from local simulation to a federated setting. Additionally, this approach offers greater control over the sequence and content of the FL experiment steps, which enables more complex experiments beyond just horizontal FL. Workflow API also provides more granular privacy controls, allowing the model owner to explicitly permit or forbid the transfer of specific attributes over the network.

### Consolidating interfaces
OpenFL has supported multiple ways of running FL experiments for a long time, many of which are not interoperable: TaskRunner API, Workflow API, Python Native API, and Interactive API. The strategic vision is to consolidate OpenFL around the Workflow API, as it focuses on meeting the needs of the data scientist, who is the main user of the framework. Over the upcoming 1.x releases, we plan to gradually deprecate and eliminate the legacy Python Native API and Interactive API. OpenFL 2.0 will be centered around the Workflow API, facilitating a seamless transition from local simulations to distributed FL experiments, and even enabling the setup of permanent federations, which is currently only possible through the Interactive API.

### Component standardization and framework interoperability

Federated Learning is a [burgeoning space](https://github.com/weimingwill/awesome-federated-learning#frameworks).
Most core FL infrastructure (model weight extraction, network protocols, and serialization designs) must be reimplemented ad hoc by each framework. 
This causes community fragmentation and distracts from some of the bigger problems to be solved in federated learning. In the short term, we want to collaborate on standards for FL, first at the communication and storage layer, and make these components modular across other frameworks. Our aim is also to provide a library for FL algorithms, compression methods, that can both be applied and interpreted easily.

### Confidential computing support
Although OpenFL currently relies on Intel® SGX for trusted execution, the long term vision is towards broader confidential computing ecosystem support. This can be achieved by packaging OpenFL workspaces and workflows as Confidential Containers (CoCo), which supports a spectrum of TEE backends, including Intel® SGX and TDX, Arm TrustZone, and AMD SEV.

## Upcoming OpenFL releases

### 1.8 (March '2025)
In this release, we intend to continue streamlining the OpenFL APIs, provide additional security options, and enhance ML/FL framework interoperability:
- Removing the Python Native API and Interactive API
- Further decoupling the Runtime from the FLSpec in Workflow API (see the design proposal [here](https://github.com/securefederatedai/openfl/discussions/1317))
- ML frameworks integration (PyTorch 2.5 support, additional Keras back-ends)
- Additional enhancements to Federated Evaluation with OpenFL, including:
  * Dynamic switching from learning to evaluation mode via TaskRunner API within the same federation, without re-distributing the FL plan
  * Workflow API tutorial for Federated Evaluation
- PoC for [Secure Aggregation](https://eprint.iacr.org/2017/281.pdf) support
- Design proposal for a configurable communication layer (enabling REST API support, in addition to gRPC)
- (TBA) A leap forward in terms of FL framework interoperability (stay tuned for announcements)

### 1.9 (TBA)
Stay tuned for updates!