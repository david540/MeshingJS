
/**
 * Create a double vector, the vector should be deleted by user
 * @param {Array} array data to copy into vector
 */
 export function CreateVectorDouble( array ){
    let vec = new Module.VectorDouble();
    for (let i = 0; i < array.length; ++i)
        vec.push_back(array[i]);
    return vec;
}

/**
 * Create an Int vector, the vector should be deleted by user
 * @param {Array} array data to copy into vector
 */
 export function CreateVectorInt( array ){
    let vec = new Module.VectorInt();
    for (let i = 0; i < array.length; ++i)
        vec.push_back(int(array[i]));
    return vec;
}

/**
 * Create an array from a vector
 * @param {VectorDouble} vector vector to extract data from
 */
 export function ExtractArray( vector ){
    let array = [];
    for (let i = 0; i < vector.size(); ++i)
        array.push(vector.get(i));
    return array;
}