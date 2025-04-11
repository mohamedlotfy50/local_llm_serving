# LLM Serving Exploration: From Scratch to Deployment

This project explores serving Large Language Models (LLMs) with a focus on understanding the entire pipeline, from initial model interaction to a fully functional deployment. The core objective is to achieve:

* **Minimal Dependencies:** Relying on a lean set of essential packages.
* **Fast Inference:** Optimizing for low-latency response generation.
* **Concurrent Requests:** Handling multiple user requests efficiently.
* **Maximum GPU Utilization (Single Instance):** Leveraging the full power of a single GPU for inference.
* **Extensibility:** Gradually expanding to support advanced functionalities:
    * Token Streaming Endpoints
    * End-to-End Voice-to-Voice Communication
    * Function Calling Capabilities
    * Multi-modality Support (Voice, Text, Images, Video)

## Technologies

The following technologies are currently being utilized in this project:

* **Flask:** A lightweight WSGI web application framework for creating the API endpoints.
* **Hugging Face Transformers:** Providing access to a wide range of pre-trained language models and utilities.
* **PyTorch:** The underlying deep learning framework for model inference and manipulation.

## Focus: Continuous Batching with Customizable Prefill

A key area of exploration in this project is **continuous batching**. This involves efficiently grouping incoming requests to maximize GPU utilization during inference. We are investigating both:

* **Continuous Batching without Prefill:** Processing requests as they arrive and batching them dynamically.
* **Continuous Batching with Prefill:** Optimizing the initial token generation (prefill) phase for batched requests.

The goal is to provide full customization over the batching strategies to adapt to different model characteristics and workload patterns.

## Architecture and Design Principles

This project emphasizes building a clean and maintainable codebase by adhering to sound architectural principles:

* **Clean Architecture:** Separating concerns into distinct layers (e.g., presentation, business logic, data access) to improve organization and testability.
* **SOLID Principles:** Applying the five SOLID principles of object-oriented design to create a robust and flexible system:
    * **Single Responsibility Principle (SRP):** Each module or class should have only one reason to change.
    * **Open/Closed Principle (OCP):** Software entities should be open for extension but closed for modification. This is particularly important for supporting different LLM models with minimal code changes.
    * **Liskov Substitution Principle (LSP):** Subtypes should be substitutable for their base types without altering the correctness of the program.
    * **Interface Segregation Principle (ISP):** Clients should not be forced to depend on interfaces they do not use.
    * **Dependency Inversion Principle (DIP):** Depend on abstractions, not on concretions.

By following these principles, the aim is to create a system that is easy to understand, modify, and extend to support new models and features.

## Educational Goals (Checklist)

The primary motivation behind this project is educational, aiming for a comprehensive understanding of the LLM lifecycle. The exploration will cover the following stages:

* [ ] **Pretraining:** Understanding the initial training process of large language models.
* [ ] **Supervised Fine-tuning (SFT):** Learning how to adapt pre-trained models for specific downstream tasks using labeled data.
* [ ] **Reinforcement Learning (RLHF or Reinforcement Learning from Human Feedback):** Exploring techniques to align model behavior with human preferences.
* [ ] **Deploying:** Packaging and preparing the model for serving in a production-like environment.
* [ ] **Serving:** Building the infrastructure and API endpoints to make the model accessible for inference.