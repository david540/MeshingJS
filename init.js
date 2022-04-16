import * as THREE from 'three';	
import { SingletonPanel } from './panel.js'
import { MeshOpe } from './mesh.js'
import { OBJLoader } from './three.js/examples/jsm/loaders/OBJLoader.js';
import { OrbitControls } from './three.js/examples/jsm/controls/OrbitControls.js';

let container;
let camera, scene, renderer;

init();
animate();

function init() {

    container = document.createElement( 'div' );
    document.body.appendChild( container );

    camera = new THREE.PerspectiveCamera( 45, window.innerWidth / window.innerHeight, 1, 2000 );
    camera.position.z = 25;

    scene = new THREE.Scene();

    const ambientLight = new THREE.AmbientLight( 0xcccccc, 0.4 );
    scene.add( ambientLight );

    const pointLight = new THREE.PointLight( 0xffffff, 0.8 );
    camera.add( pointLight );
    scene.add( camera );

    var object;
    var meshOpe;

    const manager = new THREE.LoadingManager();
    const loader = new OBJLoader( manager );
    
    loader.load( './three.js/examples/models/obj/emerald.obj', function ( obj ) {
        object = obj;
        object.position.x = 0; object.position.y = 0; object.position.z = 0;
        const mesh = object.children[0].clone();
        meshOpe = new MeshOpe(mesh);
        SingletonPanel.init(scene, meshOpe);
        meshOpe.display(scene);
    }, onProgress, onError );

    function onProgress( xhr ) {
        if ( xhr.lengthComputable ) {
            const percentComplete = xhr.loaded / xhr.total * 100;
            console.log( 'model ' + Math.round( percentComplete, 2 ) + '% downloaded' );
        }
    }
    function onError() {}

    document.querySelector('.inputfile').addEventListener('change', function(e){
        var dae_path;
        var files = e.currentTarget.files;
        if(files.length < 1) return;
        var extraFiles = {}, file;
        for (var i = 0; i < 1; i++){//files.length; i++) {
          file = files[i];
          extraFiles[file.name] = file;
          dae_path = file.name;
        }
        const manager = new THREE.LoadingManager();
        manager.setURLModifier(function (url, path) {	
          url = url.split('/');
          url = url[url.length - 1];
          if (extraFiles[url] !== undefined) {
            var blobURL = URL.createObjectURL(extraFiles[url]);
            console.log(blobURL); //Blob location created from files selected from file input
            return blobURL;	
          }return url;
        });
        const loader = new OBJLoader( manager );
        loader.load(dae_path, function ( obj ) {
            object = obj;
            scene.remove(scene.children[scene.children.length - 1]); 
            object.position.x = 0; object.position.y = 0; object.position.z = 0;
            const mesh = object.children[0].clone();
            meshOpe.reset(mesh);
            meshOpe.display(scene);
        }, onProgress, onError );           
    });

    renderer = new THREE.WebGLRenderer();
    renderer.setPixelRatio( window.devicePixelRatio );
    renderer.setSize( window.innerWidth, window.innerHeight );
    container.appendChild( renderer.domElement );
    const controls = new OrbitControls( camera, renderer.domElement );
    controls.target.set( 0, 1, 0 ); controls.update();
    window.addEventListener( 'resize', onWindowResize );
}

function onWindowResize() {
    const windowHalfX = window.innerWidth / 2;
    const windowHalfY = window.innerHeight / 2;
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize( window.innerWidth, window.innerHeight );
}
function animate() {
    requestAnimationFrame( animate );
    render();
}
function render() {
    renderer.render( scene, camera );
}
