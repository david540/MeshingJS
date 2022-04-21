# MeshingJS

## Over view
Quad mesh generation on given triangular mesh input with quadcover algorithm.

Three js permits to modify by hand the frame field topology on the input tri mesh, to see in real time the impact on the output quadrilateral mesh.

![image](https://user-images.githubusercontent.com/25902963/163992931-637d27ba-d674-492c-b73e-d3ace422a3f9.png)

## To run the project: 

Compile cpp (need wasm installation):
```bash
cd cpp
make compile
```

Run the node js server :

```bash
cd Back
npm install // if running for the first time
node server.js
```

Compile and run the page
```bash
cd Front
npm install // if running for the first time
npm run build
```
then go to localhost:8080, the page should be up 

![image](https://user-images.githubusercontent.com/25902963/163992656-9a075d12-d2e5-4de1-b0e1-2ae1b6f29571.png)

![image](https://user-images.githubusercontent.com/25902963/164543440-a6209e2f-70d5-4130-bb78-e05a3ebbcff5.png)

