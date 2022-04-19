
import * as THREE from 'three';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import {mergeVertices} from './GeometryUtils.js';
import * as interfaceUtils from './interfaceUtils.js';
import { VertexNormalsHelper } from 'three/examples/jsm/helpers/VertexNormalsHelper.js';

class Mesh {
    constructor(path = './examples/emerald.obj') {
        this.filePath = path
        this.object = new THREE.Group()
        this.mesh = null
        this.computation_mesh = null
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
            this.computation_mesh = this.mesh.clone()
            this.computation_mesh.geometry = mergeVertices(this.mesh.geometry, 1e-8)
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
        const loader = new THREE.TextureLoader();
        const texture = loader.load( '../examples/disc.png' );
        const pointsMaterial = new THREE.PointsMaterial( {
            color: new THREE.Color( 255, 0, 0 ),
            map: texture,
            size: 1,
            alphaTest: 0.5
        } );
        this.pointsCloud = new THREE.Points(this.mesh.geometry, pointsMaterial)
        this.pointsCloud.name = "pointsCloud";
        this.object.add(this.pointsCloud);
    }

    updateDisplay(displayOptions) {
        this.pointsCloud.visible = displayOptions.bIsShowVerts
        this.wireframe.visible = displayOptions.bIsShowEdges
        this.pointsCloud.material.size = displayOptions.pointWidth
    }

    extractVertices() {
        if(!this.mesh) return []
        return interfaceUtils.CreateVectorDouble(this.computation_mesh.geometry.getAttribute("position").array);
    }

    extractFaceIndices() {
        if(!this.mesh) return []
        return interfaceUtils.CreateVectorInt(this.computation_mesh.geometry.index.array);
    }

    computeFF() {
        let data = interfaceUtils.ExtractArray(Module.computeFF(this.extractFaceIndices(), this.extractVertices()))
        const dir_arr = data.slice(0, data.length/2).concat(data.slice(0, data.length/2).map(function(x) {return -x}));
        const vert_arr = data.slice(data.length/2).concat(data.slice(data.length/2))
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute( 'position', new THREE.BufferAttribute( new Float32Array(vert_arr), 3 ) );
        geometry.setAttribute( 'normal', new THREE.BufferAttribute( new Float32Array(dir_arr), 3 ) );
        const material = new THREE.MeshBasicMaterial( { color: 0xff0000 } );
        const mesh = new THREE.Mesh( geometry, material )
        let helper = new VertexNormalsHelper( mesh, 1, new THREE.Color(255, 0, 0), 3 );
        helper.name = "frameField"
        this.object.add(helper)
    }
}

export { Mesh };
