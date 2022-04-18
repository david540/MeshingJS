import { terser } from "rollup-plugin-terser"; // code minification (optional)
import commonjs from '@rollup/plugin-commonjs';
import resolve from '@rollup/plugin-node-resolve'; // locate and bundle dependencies in node_modules (mandatory)

export default {
	input: 'js/Scene.js',
	output: [
		{
			format: 'umd',
			name: 'MP',
			file: 'build/MeshProcessing.js'
		}
	],
	onwarn: function(warning) {
		if ( warning.code === 'THIS_IS_UNDEFINED' ) { return; }// Skip TWEEN WARNING
		console.warn( warning.message );// console.warn everything else
	},
	plugins: [resolve(), commonjs()]
};
