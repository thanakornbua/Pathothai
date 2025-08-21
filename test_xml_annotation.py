import os
from scripts.train import WholeSlideDataset

def test_xml_parsing():
    # Example XML file and dummy slide
    xml_path = os.path.join('Annotations', 'Her2Neg_Case_01.xml')
    slide_path = 'data\SVS\Her2Neg_Case_01.svs'  # Use a real image if available
    labels = ['Tumor']
    annotation_paths = [xml_path]
    dataset = WholeSlideDataset(
        slide_paths=[slide_path],
        labels=labels,
        patch_size=1000,
        patches_per_slide=50,
        annotation_paths=annotation_paths
    )
    print('Parsed regions:', dataset.regions)
    # Try extracting a patch (will fail if slide is not present, but should parse XML)
    try:
        patch, label = dataset[0]
        print('Patch shape:', patch.size, 'Label:', label)
    except Exception as e:
        print('Patch extraction error (expected if slide missing):', e)

if __name__ == '__main__':
    test_xml_parsing()
