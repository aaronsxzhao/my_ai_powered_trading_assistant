-- Brooks Trading Coach - Storage Policies
-- Run this after creating the 'materials' bucket in Supabase Dashboard
-- ============================================================

-- NOTE: First, create the bucket manually in Supabase Dashboard:
-- Storage > New Bucket > Name: "materials" > Private: checked

-- ============================================================
-- Storage policies for materials bucket
-- ============================================================

-- Allow users to upload to their own folder
-- Files are stored as: materials/{user_id}/{filename}
CREATE POLICY "Users can upload to own folder" ON storage.objects
    FOR INSERT 
    WITH CHECK (
        bucket_id = 'materials' AND
        (storage.foldername(name))[1] = auth.uid()::text
    );

-- Allow users to read their own files
CREATE POLICY "Users can read own files" ON storage.objects
    FOR SELECT 
    USING (
        bucket_id = 'materials' AND
        (storage.foldername(name))[1] = auth.uid()::text
    );

-- Allow users to update their own files
CREATE POLICY "Users can update own files" ON storage.objects
    FOR UPDATE
    USING (
        bucket_id = 'materials' AND
        (storage.foldername(name))[1] = auth.uid()::text
    );

-- Allow users to delete their own files
CREATE POLICY "Users can delete own files" ON storage.objects
    FOR DELETE 
    USING (
        bucket_id = 'materials' AND
        (storage.foldername(name))[1] = auth.uid()::text
    );
