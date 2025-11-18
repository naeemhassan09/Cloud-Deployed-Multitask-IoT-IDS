# ECS Service Configuration (Notes)

- Cluster: <your-ecs-cluster-name>
- Service: iot-multitask-ids-api
- Launch type: FARGATE
- Desired count: 1–2 tasks
- Load balancer: Application Load Balancer
- Target group: HTTP 80 -> container port 8000
