
import * as THREE from 'three';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';

class Mesh {
    constructor(path = './examples/emerald.obj') {
        this.filePath = path
        this.object = new THREE.Group()
        this.mesh = null
        this.wireframe = null
        this.pointsCloud = null
    }

    async loadMesh() {
        this.object = new THREE.Group()

        //Load object
        const loadingManager = new THREE.LoadingManager();
        const objLoader = new OBJLoader(loadingManager);
        await objLoader.load(this.filePath, ((obj) => {
            obj.position.fromArray([0, 0, 0]);
            obj.name = "mesh"
            this.mesh = obj.children[0]
            this.object.add(this.mesh);

            //set wireframe
            this.wireframe = new THREE.LineSegments(new THREE.WireframeGeometry(this.mesh.geometry), new THREE.LineBasicMaterial({ color: 0x000000, linewidth: 2 }));
            this.wireframe.name = "wireframe"
            this.object.add(this.wireframe)

            //set points cloud
            this.computePointsCloud()
        }).bind(this));

    }

    computePointsCloud() {
        let pos = this.mesh.geometry.attributes.position;
        var colors = []; for (var i = 0; i < pos.count; i++) colors.push(255, 0, 0);
        const geom_pt = new THREE.BufferGeometry();
        geom_pt.setAttribute('position', pos); geom_pt.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        const mt_pt = new THREE.PointsMaterial({ vertexColors: true, size: 1, sizeAttenuation: true, alphaTest: 0.5, transparent: true });
        this.pointsCloud = new THREE.Points(geom_pt, mt_pt);
        this.pointsCloud.name = "pointsCloud";
        this.object.add(this.pointsCloud);
    }


    updateDisplay(displayOptions) {
        this.pointsCloud.visible = displayOptions.bIsShowVerts
        this.wireframe.visible = displayOptions.bIsShowEdges
        this.pointsCloud.material.size = displayOptions.pointWidth
    }


    computeFF() {
        console.log("Mesh indexes : ")
        console.log(this.mesh.geometry.index);
    }
}

export { Mesh };
