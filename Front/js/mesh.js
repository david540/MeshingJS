
import * as THREE from 'three';
import { SingletonPanel } from './panel.js'
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';

class Mesh {
    constructor(path = './examples/emerald.obj') {
        this.filePath = path
        this.object = new THREE.Group()
        this.mesh = null
        this.wireframe = null
        this.pointsCloud = null
    }

    loadMesh() {
        this.object = new THREE.Group()

        //Load object
        const loadingManager = new THREE.LoadingManager();
        const objLoader = new OBJLoader(loadingManager);
        objLoader.load(this.filePath, ((obj) => {
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

            //SingletonPanel.init(scene, meshOpe);
            //meshOpe.display(scene);
        }).bind(this));

    }

    computePointsCloud() {
        let pos = this.mesh.geometry.attributes.position;
        var colors = []; for (var i = 0; i < pos.count; i++) colors.push(255, 0, 0);
        const geom_pt = new THREE.BufferGeometry();
        geom_pt.setAttribute('position', pos); geom_pt.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        const mt_pt = new THREE.PointsMaterial({ vertexColors: true, size: SingletonPanel.point_width, sizeAttenuation: true, alphaTest: 0.5, transparent: true });
        this.pointsCloud = new THREE.Points(geom_pt, mt_pt);
        this.pointsCloud.name = "pointsCloud";
        this.object.add(this.pointsCloud);
    }


    display(scene) {
        const cp_mesh = this.mesh.clone();
        cp_mesh.material.color = { r: 0.4, g: 0.4, b: 0.4 };
        scene.add(cp_mesh);
        if (SingletonPanel.b_show_edges) cp_mesh.add(this.wireframe);
        if (SingletonPanel.b_show_verts) {
            this.points.material.size = SingletonPanel.point_width;
            cp_mesh.add(this.points);
        }
    }


    computeFF() {
        console.log("Mesh indexes : ")
        console.log(this.mesh.geometry.index);
    }
}

export { Mesh };
