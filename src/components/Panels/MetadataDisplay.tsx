/**
 * Metadata Display Component
 *
 * Displays DICOM metadata in a structured format.
 */

import { useDicomStore } from '../../stores';

export function MetadataDisplay() {
  const metadata = useDicomStore((s) => s.metadata);
  const isLoaded = useDicomStore((s) => s.isLoaded);

  if (!isLoaded || !metadata) {
    return (
      <div className="text-gray-400 text-sm p-4 text-center">
        No DICOM file loaded
      </div>
    );
  }

  const sections = [
    {
      title: 'Patient',
      items: [
        { label: 'Patient ID', value: metadata.patientId },
        { label: 'Patient Name', value: metadata.patientName },
      ],
    },
    {
      title: 'Study',
      items: [
        { label: 'Study Date', value: metadata.studyDate },
        { label: 'Study Description', value: metadata.studyDescription },
        { label: 'Modality', value: metadata.modality },
        { label: 'Manufacturer', value: metadata.manufacturer },
      ],
    },
    {
      title: 'Image',
      items: [
        { label: 'Dimensions', value: `${metadata.rows} x ${metadata.columns}` },
        { label: 'Frames', value: metadata.numFrames?.toString() },
        { label: 'Frame Rate', value: metadata.frameRate ? `${metadata.frameRate} fps` : undefined },
        { label: 'Pixel Spacing', value: metadata.pixelSpacing ? `${metadata.pixelSpacing.toFixed(4)} mm/px` : undefined },
      ],
    },
    {
      title: 'Acquisition',
      items: [
        { label: 'Primary Angle', value: metadata.primaryAngle },
        { label: 'Secondary Angle', value: metadata.secondaryAngle },
      ],
    },
  ];

  return (
    <div className="space-y-4 text-sm">
      {sections.map((section) => {
        const filteredItems = section.items.filter((item) => item.value);
        if (filteredItems.length === 0) return null;

        return (
          <div key={section.title}>
            <h4 className="text-gray-300 font-medium mb-2 border-b border-gray-700 pb-1">
              {section.title}
            </h4>
            <div className="space-y-1">
              {filteredItems.map((item) => (
                <div key={item.label} className="flex justify-between">
                  <span className="text-gray-400">{item.label}</span>
                  <span className="text-white">{item.value}</span>
                </div>
              ))}
            </div>
          </div>
        );
      })}

      {/* ECG Status */}
      <div>
        <h4 className="text-gray-300 font-medium mb-2 border-b border-gray-700 pb-1">
          ECG
        </h4>
        <div className="flex justify-between">
          <span className="text-gray-400">ECG Data</span>
          <span className={metadata.hasEcg ? 'text-green-400' : 'text-gray-500'}>
            {metadata.hasEcg ? 'Available' : 'Not available'}
          </span>
        </div>
      </div>
    </div>
  );
}
